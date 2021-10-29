use std::time::{Duration, Instant};

use rand::prelude::*;
use rand_distr::{UnitSphere, Distribution};
use ndarray::{Array3,Array4};
use ndarray::Array;
use ndarray::prelude::*;
use ndrustfft::{Complex};
use pyo3::prelude::*;
use crate::fft_convolution::{fft, convolve};
use crate::tumor_model;

pub type Real = f32;
pub const DIM: usize = 3; //not intended to work in other than 3 dimensions, just for code clarity
pub const MAX_INDEX: usize = 256;

const UNIT_SPHERE_VOL: Real = 4 as Real*(std::f64::consts::PI as Real)/3 as Real;

#[derive(Copy, Clone, PartialEq)]
pub enum CellType {
	Neuron = 0,
	Cancer = 1,
	Stroma = 2,
	Microglia = 3,
}

#[derive(Clone)]
pub struct Cell {
	pub(crate) x: Real,
	pub(crate) y: Real,
	pub(crate) z: Real,
	pub(crate) cell_type: CellType,
}

//Contains the biological parameters that set the rules for a tumor model
#[pyclass]
#[derive(Clone)]
pub struct TumorModel {
	cell_effective_radii: [Real; 3],
	cancer_growth: Real,
	immune_growth: Real,
	healthy_cancer_diffusion: Real,
	cytotoxicity: Real,
	neighbor_distance: Real,
}

#[pymethods]
impl TumorModel {
	#[new]
	fn new(cell_effective_radii: [Real; 3]) -> Self {
		TumorModel{cell_effective_radii,
			cancer_growth: 1.0,
			immune_growth: 0.1,
			healthy_cancer_diffusion: 5.0,
			cytotoxicity: 0.5,
			neighbor_distance: 20.,
		}
	}
}

//Holds memory needed for running tumor simulation
pub struct SimulationAllocation {
	expansion: Array3<Real>,
	expansion_hat: Array3<Complex<Real>>,
	displacement: [Array3<Real>; DIM],
	displacement_kernel_hat: [Array3<Complex<Real>>; DIM],
	temp: Array3<Complex<Real>>, //storage used during fft
	index: Array4<usize>,
	index_n: Array3<usize>,
	dx: Real,
	n: usize,
}

impl SimulationAllocation {
	pub(crate) fn get_grid_i(&self, p0: Real, p1: Real, p2: Real) -> (i32, i32, i32) {
		return ((p0/self.dx) as i32, (p1/self.dx) as i32, (p2/self.dx) as i32)
	}
}

//Contains state of tumor
pub struct Tumor {
	model: TumorModel,
	pub(crate) cells: Vec<Cell>,
	size: Real,
	rng: ThreadRng,
}

impl Tumor {
	pub(crate) fn count_cells(&self, cell_type : CellType) -> usize {
		self.cells.iter().filter(|&c| c.cell_type == cell_type).count()
	}
}

fn get_initial_tumor(model: TumorModel, size: Real, n_neurons: usize, n_stroma: usize, n_microglia: usize) -> Tumor {
	let mut rng = rand::thread_rng();
	let mut cells = Vec::new();
	let half = size/2.0;
	for _ in 0..n_neurons {
		cells.push(Cell {x:rng.gen::<Real>()*size, y:rng.gen::<Real>()*size, z:rng.gen::<Real>()*size, cell_type:CellType::Neuron });
	}
	for _ in 0..n_stroma {
		cells.push(Cell {x:rng.gen::<Real>()*size, y:rng.gen::<Real>()*size, z:rng.gen::<Real>()*size, cell_type:CellType::Stroma });
	}
	for _ in 0..n_microglia {
		cells.push(Cell {x:rng.gen::<Real>()*size, y:rng.gen::<Real>()*size, z:rng.gen::<Real>()*size, cell_type:CellType::Microglia });
	}
	cells.push(Cell {x:half, y:half, z:half, cell_type:CellType::Cancer});
	Tumor{model, size, cells, rng }
}

fn initialize_simulation(tumor: &Tumor, resolution: Real, n_threads: u32) -> SimulationAllocation{
	let min_n = (tumor.size / resolution) as usize;
	//Find first power of 2 large enough (replace with better approx later since we can have more prime factors. This is a huge performance gain since dimensionality is high)
	let mut m=1 as usize;
	let n_fft = loop { if m >= 2*min_n - 1 {break m}; m*=2;};
	let n = (n_fft+1)/2; //as large n as possible
	let n_kernel = (2*n-1) as usize;
	let dx = tumor.size / (n as Real);

	// let kernel_xs = Array::<Real, Ix1>::linspace(-((n-1) as Real)*dx, ((n-1) as Real)*dx, 2*n-1);
	let kernel_xs = Array::from_iter((0..n_kernel).map(|i| (((i + n)%n_kernel) as i32 - n as i32) as Real * dx));
	let mut kernel_coords = Array::<Real, Ix4>::zeros((n_kernel, n_kernel, n_kernel, DIM));
	for i in 0..kernel_xs.len() //make dim general
	{
		for j in 0..kernel_xs.len()
		{
			for k in 0..kernel_xs.len()
			{
				kernel_coords[[i,j,k,0]] = kernel_xs[i]; //make loop :)
				kernel_coords[[i,j,k,1]] = kernel_xs[j];
				kernel_coords[[i,j,k,2]] = kernel_xs[k];
			}
		}
	}

	// let mut displacement_kernel = Array::zeros((n_kernel, n_kernel, n_kernel, DIM));
	let mut temp = Array3::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel));

	let mut max_r = 0 as Real;
	for r in tumor.model.cell_effective_radii{
		if r > max_r{
			max_r = r;
		}
	}
	let kernel_rs = kernel_coords.map(|x| x*x).sum_axis(Axis(DIM)).map(|r2| r2.sqrt());
	let middle = n-1;
	let max_vol = max_r.powi(3) * UNIT_SPHERE_VOL;
	let mut f = kernel_rs.map(|r| (((1 as Real - (-(r/max_r).powi(3)).exp_m1()*((max_r / r).powi(3)) ).powf(  (DIM as Real).recip()  ) - 1 as Real)/max_vol));
	//127 = 2*64-1
	f[[0, 0, 0]] = 0 as Real;
	// let displacement_kernel: [Array3<Real>;3] = [Array3::<Real>::zeros((n,n,n)), Array3::<Real>::zeros((n,n,n)), Array3::<Real>::zeros((n,n,n))];
	let mut displacement_kernel_hat: [Array3<Complex<Real>>;3] = [Array3::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel)), Array3::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel)), Array3::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel))];
	for i in 0..DIM{
		fft(&(&f*kernel_coords.index_axis(Axis(DIM), i).to_owned()),
			&mut displacement_kernel_hat[i],
			&mut temp);
		// println!("displacement_kernel_hat[i][[0,0,0]]: {}", displacement_kernel_hat[i][[0,0,0]]);
	}
	
	let index_n = (tumor.size / tumor.model.neighbor_distance) as usize;

	SimulationAllocation {
		expansion: Array::zeros((n_kernel, n_kernel, n_kernel)),
		expansion_hat: Array3::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel)),
		displacement: [Array3::<Real>::zeros((n_kernel, n_kernel, n_kernel)), Array3::<Real>::zeros((n_kernel, n_kernel, n_kernel)), Array3::<Real>::zeros((n_kernel, n_kernel, n_kernel))],
		displacement_kernel_hat,
		temp,
		index: Array::zeros((index_n, index_n, index_n, MAX_INDEX)),
		index_n: Array::zeros((index_n, index_n, index_n)),
		dx,
		n,
	}
}

// cell index is saved in all voxels that overlap with sphere centered at cell with interaction dist radius
// This way we only need to look in one 
fn populate_index(tumor: &mut Tumor, all: &mut SimulationAllocation){
	//add cell ids in spatial grid
	//create index in parallel
	
	let n = all.index_n.shape()[0] as i32;
	let rad = tumor.model.neighbor_distance;
	all.index_n.fill(0);
	let mut overflows = 0 as usize;

	for i in 0..tumor.cells.len(){
		//let mut corner_indices: Vec<[usize; DIM]>;
		// let mut corner = [bool; DIM];
		let (px, py, pz) = (tumor.cells[i].x, tumor.cells[i].y, tumor.cells[i].z);
		let mut prev_xj = -1 as i32;
		for xi in 0..2{
			let xj = ((px + rad*(2*xi-1) as Real)/all.dx) as i32;
			if xj<0 || xj >= n {continue;}
			if prev_xj != xj {
				prev_xj = xj;
				let mut prev_yj = -1 as i32;
				for yi in 0..2{
					let yj = ((py + rad*(2*yi-1) as Real)/all.dx) as i32;
					if yj<0 || yj >= n {continue;}
					if prev_yj != yj{
						prev_yj = yj;
						let mut prev_zj = -1 as i32;
						for zi in 0..2{
							let zj = ((pz + rad*(2*zi-1) as Real)/all.dx) as i32;
							if zj<0 || zj >= n {continue;}
							if prev_zj != zj{
								prev_zj = zj;
								if all.index_n[[xj as usize, yj as usize, zj as usize]] < MAX_INDEX {
									all.index[[xj as usize, yj as usize, zj as usize, all.index_n[[xj as usize, yj as usize, zj as usize]]]] = i;
									all.index_n[[xj as usize, yj as usize, zj as usize]] += 1;
								}
								else
								{
									overflows += 1;
								}
							}
						}
					}
				}
			}
		}
	}
	if overflows > 0
	{
		println!("{} index overflows!",overflows);
	}
}

fn remove_outside_cells(tumor: &mut Tumor){
	let s = tumor.size;
	tumor.cells.retain(|c| c.x >= 0. && c.x <= s && c.y >= 0. && c.y <= s && c.z >= 0. && c.z <= s);
}

fn split_cells(tumor: &mut Tumor, all: &mut SimulationAllocation, dt: Real){
	for i in 0..tumor.cells.len() {
		match tumor.cells[i].cell_type {
			CellType::Cancer => {
				if tumor.rng.gen::<Real>() < tumor.model.cancer_growth*dt {
					split_cell(tumor,all,i);
				}
			}
			CellType::Stroma => {
				if tumor.rng.gen::<Real>() < tumor.model.immune_growth*dt {
					split_cell(tumor,all,i);
				}
			}
			_ => (),
		}
	}
}

fn split_cell(tumor: &mut Tumor, all: &mut SimulationAllocation, i:usize){
	let xi = ((tumor.cells[i].x as Real)/all.dx) as usize;
	let yi = ((tumor.cells[i].y as Real)/all.dx) as usize;
	let zi = ((tumor.cells[i].z as Real)/all.dx) as usize;
	let r = tumor.model.cell_effective_radii[tumor.cells[i].cell_type as usize];
	let vol = UNIT_SPHERE_VOL*r*r*r;
	all.expansion[[xi,yi,zi]] += vol;
	tumor.cells.push(tumor.cells[i].clone());
	
	let v: [Real; 3] = UnitSphere.sample(&mut tumor.rng);
	let d = r;
	tumor.cells[i].x -= d*v[0];
	tumor.cells[i].y -= d*v[1];
	tumor.cells[i].z -= d*v[2];
	tumor.cells.last_mut().unwrap().x += d*v[0];
	tumor.cells.last_mut().unwrap().y += d*v[1];
	tumor.cells.last_mut().unwrap().z += d*v[2];
}

fn displace_cells(tumor: &mut Tumor, all: &mut SimulationAllocation){
	fft(&all.expansion, &mut all.expansion_hat, &mut all.temp);
	for i in 0..DIM {
		convolve(&all.displacement_kernel_hat[i],
				 &all.expansion_hat,
				 &mut all.displacement[i],
				 &mut all.temp);
	}
	all.expansion.fill(0 as Real);
	for c in & mut tumor.cells{
		let (ix, iy, iz) = all.get_grid_i(c.x, c.y, c.z);
		if ix>=0 && ix< all.n as i32 && iy>=0 && iy<all.n as i32 && iz>=0 && iz<all.n as i32{
			// println!("all.displacement[0][[ix as usize,iy as usize,iz as usize]]: {}", all.displacement[0][[ix as usize,iy as usize,iz as usize]]);
			c.x += all.displacement[0][[ix as usize,iy as usize,iz as usize]];
			c.y += all.displacement[1][[ix as usize,iy as usize,iz as usize]];
			c.z += all.displacement[2][[ix as usize,iy as usize,iz as usize]];
		}
	}
}

fn diffuse_cells(tumor: &mut Tumor, all: &mut SimulationAllocation){
	for c in & mut tumor.cells{

	}
}

fn find_pairs(tumor: &mut Tumor, all: &mut SimulationAllocation){
	let pairs = Vec::new();

}

fn update_tumor(tumor: &mut Tumor, all: &mut SimulationAllocation, dt: Real){
	let start = Instant::now();
	populate_index(tumor, all);
	println!("populate_index: {} ms", start.elapsed().as_millis());

	let start = Instant::now();
	split_cells(tumor, all, dt);
	println!("split_cells: {} ms", start.elapsed().as_millis());

	let start = Instant::now();
	displace_cells(tumor, all);
	println!("displace_cells: {} ms", start.elapsed().as_millis());

	let start = Instant::now();
	diffuse_cells(tumor, all);
	println!("diffuse_cells: {} ms", start.elapsed().as_millis());

	// let start = Instant::now();
	// small_scale_displace_cells(tumor, all);
	// println!("small_scale_displace_cells: {} ms", start.elapsed().as_millis());

	let start = Instant::now();
	remove_outside_cells(tumor);
	println!("remove_outside_cells: {} ms", start.elapsed().as_millis());
}

pub(crate) fn simulate(model:TumorModel, size: Real, n_heathy: usize, n_stroma: usize, n_microglia: usize, max_tumor_cells: usize) -> Tumor{
	let mut tumor = get_initial_tumor(model, size, n_heathy, n_stroma, n_microglia);
	let resolution = 12.;
	let n_threads = 4;

	println!("Initializing simulation..");
	let mut allocation = initialize_simulation(&tumor, resolution, n_threads);

	let dt = 0.1;
	loop {
		println!("Number of cells: {}", tumor.cells.len());
		if tumor.count_cells(CellType::Cancer) >= max_tumor_cells{
			break;
		}
		update_tumor(&mut tumor,&mut  allocation, dt);
	}
	tumor
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn simulate_tumor() {
		let model = TumorModel::new([10.,10.,10.]);
		let min_cell_number = 100;
		let tumor = simulate(model, 100., 10, 10, 10, min_cell_number);
		assert!(tumor.count_cells(CellType::Cancer) >= min_cell_number, "Not enough cells at end of simulation.");
	}
}