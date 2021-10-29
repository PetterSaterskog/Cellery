use rand::prelude::*;
use rand_pcg::Pcg64;
use rand_distr::{UnitSphere, Distribution, Normal};
use ndarray::{Array3,Array4};
use ndarray::Array;
use ndarray::prelude::*;
use ndrustfft::{ndfft_r2c_par, ndifft_r2c_par, ndfft_par, ndifft_par, Complex, R2cFftHandler, FftHandler};
use approx::{assert_relative_eq};

type Real = f32;
const DIM: usize = 3; //not intended to work in other than 3 dimensions, just for code clarity
const MAX_INDEX: usize = 256;

#[derive(Copy, Clone)]
enum CellType {
	Healthy = 0,
	Cancer = 1,
	Immune = 2,
}

#[derive(Clone)]
struct Cell {
	x: Real,
	y: Real,
	z: Real,
	cell_type: CellType,
}

//Contains the biological parameters that set the rules for a tumor model
struct TumorModel {
	cell_effective_radii: [Real; 3],
	cancer_growth: Real,
	immune_growth: Real,
	healthy_cancer_diffusion: Real,
	cytotoxicity: Real,
	neighbor_distance: Real,
}

//Holds memory needed for running tumor simulation
struct SimulationAllocation {
	expansion: Array3<Real>,
	expansion_hat: Array3<Complex<Real>>,
	displacement: Array4<Real>,
	displacement_kernel_hat: Array4<Complex<Real>>,
	temp: Array3<Complex<Real>>, //storage used during fft
	index: Array4<usize>,
	index_n: Array3<usize>,
	dx: Real,
}

//Contains state of tumor
struct Tumor {
	model: TumorModel,
	cells: Vec<Cell>,
	size: Real,
	rng: ThreadRng,
}

fn read_tumor_model(_fn: String) -> TumorModel{
	TumorModel{cell_effective_radii:[10.0, 10.0, 10.0],
		cancer_growth: 1.0,
		immune_growth: 0.1,
		healthy_cancer_diffusion: 5.0,
		cytotoxicity: 0.5,
		neighbor_distance: 20.,
	}
}

fn get_initial_tumor(model: TumorModel, size: Real, immune_ratio: Real) -> Tumor {
	let mut vec = Vec::new();
	let half = size/2.0;
	let n_healthy = 0;
	for i in 0..n_healthy{
		let c = Cell {x:half, y:half, z:half, cell_type:CellType::Healthy};
		vec.push(c);
	}
	vec.push(Cell {x:half, y:half, z:half, cell_type:CellType::Cancer});
	Tumor{model, size:size, cells:vec, rng:rand::thread_rng()}
}

fn initialize_simulation(tumor: &Tumor, resolution: Real, n_threads: u32) -> SimulationAllocation{
	let min_n = (tumor.size / resolution) as usize;
	//Find first power of 2 large enough (replace with better approx later since we can have more prime factors. This is a huge performance gain since dimensionality is high)
	let mut m=1 as usize;
	let n_fft = loop { if m >= 2*min_n - 1 {break m}; m*=2;};
	let n = (n_fft+1)/2; //as large n as possible
	let n_kernel = (2*n-1) as usize;
	let dx = tumor.size / (n as Real);

	let kernel_xs = Array::<Real, Ix1>::linspace(-((n-1) as Real)*dx, ((n-1) as Real)*dx, 2*n-1);
	let mut kernel_coords = Array::<Real, Ix4>::zeros((n_kernel, n_kernel, n_kernel, n_kernel));
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
	for i in 0..DIM
	{
		for j in 0..kernel_xs.len()
		{
			let mut slice = kernel_coords.index_axis_mut(Axis(i), j);
			slice.fill(kernel_xs[j]);
		}
	}

	let radii2 = kernel_coords.mapv(|x| x.powi(2)).map_axis(Axis(3), |x| x.sum());

	let mut displacement_kernel = Array::zeros((n_kernel, n_kernel, n_kernel, DIM));
	let mut temp = Array3::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel));
	
	displacement_kernel = kernel_coords.clone();
	
	let mut displacement_kernel_hat = Array4::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel, DIM));
	
	for i in 0..DIM
	{
		fft(&displacement_kernel.index_axis_mut(Axis(DIM), i).to_owned(),
			&mut displacement_kernel_hat.index_axis_mut(Axis(DIM), i).to_owned(),
			&mut temp);
	}
	
	let index_n = (tumor.size / tumor.model.neighbor_distance) as usize;

	SimulationAllocation {
		expansion: Array::zeros((n_kernel, n_kernel, n_kernel)),
		expansion_hat: Array3::<Complex<Real>>::zeros((n_kernel/2+1, n_kernel, n_kernel)),
		displacement: Array::zeros((n_kernel, n_kernel, n_kernel, DIM)),
		displacement_kernel_hat: displacement_kernel_hat,
		temp: temp,
		index: Array::zeros((index_n, index_n, index_n, MAX_INDEX)),
		index_n: Array::zeros((index_n, index_n, index_n)),
		dx: dx,
	}
}

// #fft of the last d axes
// 	def FFT(self, f):
// 		return fft.rfftn(f, s = self.d*(self.fftSize,))

// #last indices of arguments correspond to spatial grid postions.
// def convolve(self, aFFT, bFFT):
// 	return fft.irfftn( aFFT*bFFT, s = self.d*(self.fftSize,) )[(np.s_[...],) + self.d*(np.s_[self.gridN-1:2*self.gridN-1],)]

fn fft(a: &Array3::<Real>, a_hat: & mut Array3::<Complex<Real>>,  temp: &mut Array3::<Complex<Real>>){ //a: Array::<f32, Ix3>){
	// println!("allocating..");
	// let (nx, ny, nz) = (400, 400, 400);
	// let mut data = Array3::<Real>::zeros((nx, ny, nz));
	// // let mut vhat = Array3::<Complex<Real>>::zeros((nx / 2 + 1, ny, nz));
	// for (i, v) in data.iter_mut().enumerate() {
	// 	*v = (i%23) as Real;
	// }
	// let data_c = data.clone();
	// assert_eq!(a.shape()[0]/2+1, a_hat.shape()[0], "asdf");
	// assert_eq!(a.shape()[1], a_hat.shape()[1], "asdf");
	// assert_eq!(a.shape()[2], a_hat.shape()[2], "asdf");

	// assert_eq!(a.shape()[0]/2+1, temp.shape()[0], "asdf");
	// assert_eq!(a.shape()[1], temp.shape()[1], "asdf");
	// assert_eq!(a.shape()[2], temp.shape()[2], "asdf");

	let nx = a.shape()[0];
	let mut rfft_handler = R2cFftHandler::<Real>::new(nx);
	let mut fft_handler = FftHandler::<Real>::new(nx);

	// println!("transform..");
	// println!("a.shape() {}, {}, {}", a.shape()[0], a.shape()[1], a.shape()[2]);
	// println!("a_hat.shape() {}, {}, {}", a_hat.shape()[0], a_hat.shape()[1], a_hat.shape()[2]);
	// println!("temp.shape() {}, {}, {}", temp.shape()[0], temp.shape()[1], temp.shape()[2]);
	ndfft_r2c_par(
		a,
		a_hat,
		&mut rfft_handler,
		0,
	);
	ndfft_par(
		a_hat,
		temp,
		&mut fft_handler,
		1,
	);
	ndfft_par(
		temp,
		a_hat,
		&mut fft_handler,
		2,
	);
	// println!("inverse..");
	// let mut ifft_handler = R2cFftHandler::<Real>::new(nx);
	// ndifft_r2c_par(
	// 	&mut vhat.view_mut(),
	// 	&mut data.view_mut(),
	// 	&mut ifft_handler,
	// 	0,
	// );
	// println!("{},{},{},{}", data_c[[0,0,0]], data_c[[0,0,1]], data_c[[0,0,2]], data_c[[0,0,3]]);
	// println!("{},{},{},{}", data[[0,0,0]], data[[0,0,1]], data[[0,0,2]], data[[0,0,3]]);
}

// Convolve a * b -> c
fn convolve(a_hat: &Array::<Complex<Real>, Ix3>, b_hat: &Array::<Complex<Real>, Ix3>, c: &mut Array::<Real, Ix3>, temp: &mut Array::<Complex<Real>, Ix3>)
{
	let nx = a_hat.shape()[0]*2-1;
	let ny = a_hat.shape()[1];
	let nz = a_hat.shape()[2];
	assert_eq!(nx, ny);
	assert_eq!(nx, nz);

	let mut ab_hat = a_hat * b_hat;
	// println!("inverse..");
	let mut irfft_handler = R2cFftHandler::<Real>::new(nx);
	let mut ifft_handler = FftHandler::<Real>::new(nx);

	ndifft_par(
		&ab_hat,
		temp,
		&mut ifft_handler,
		2,
	);

	ndifft_par(
		temp,
		&mut ab_hat,
		&mut ifft_handler,
		1,
	);

	ndifft_r2c_par(
		&ab_hat,
		c,
		&mut irfft_handler,
		0,
	);
}

// fn posToGridInd(x: Reals){

// }

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
		let mut corner_indices: Vec<[usize; DIM]>; 
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
	// for i in 0..tumor.cells.len(){
	// 	if tumor.cells[i].x < 0. || tumor.cells[i].x > tumor.size ||
	// 	   tumor.cells[i].y < 0. || tumor.cells[i].y > tumor.size ||
	// 	   tumor.cells[i].z < 0. || tumor.cells[i].z > tumor.size {
	// 		let last = tumor.cells.pop().unwrap();
	// 		if i<tumor.cells.len(){
	// 			tumor.cells[i] = last;
	// 			println!("removed");
	// 		}
	// 	}
	// }
}

fn split_cells(tumor: &mut Tumor, all: &mut SimulationAllocation, dt: Real){
	for i in 0..tumor.cells.len() {
		match tumor.cells[i].cell_type {
			CellType::Cancer => {
				if tumor.rng.gen::<Real>() < tumor.model.cancer_growth*dt {
					split_cell(tumor,all,i);
				}
			}
			CellType::Immune => {
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
	let vol = 4.*(std::f64::consts::PI as Real)*r*r*r/3.;
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
	for i in 0..DIM{
		convolve(&all.displacement_kernel_hat.index_axis_mut(Axis(DIM), i).to_owned(),
		 &all.expansion_hat,
		 &mut all.displacement.index_axis_mut(Axis(DIM), i).to_owned(),
		 &mut all.temp);
	}
	//displace cells
}

fn update_tumor(tumor: &mut Tumor, all: &mut SimulationAllocation, dt: Real){
	populate_index(tumor, all);
	split_cells(tumor, all, dt);
	displace_cells(tumor, all);
	remove_outside_cells(tumor);
}

fn save_tumor(tumor: &Tumor){
	
}

fn simulate(model:TumorModel, max_tumor_cells: usize) -> Tumor{
	let mut tumor = get_initial_tumor(model, 200.0, 0.5);
	let resolution = 12.;
	let n_threads = 4;

	println!("Initializing simulation..");
	let mut allocation = initialize_simulation(&tumor, resolution, n_threads);

	let dt = 0.1;
	loop {
		
		println!("Number of cells: {}", tumor.cells.len());
		if tumor.cells.len() >= max_tumor_cells{
			break;
		}
		update_tumor(&mut tumor,&mut  allocation, dt);
		
	}
	tumor
}

fn main() {
	let model = read_tumor_model(String::from("ModelFile"));
	simulate(model, 1_000_000);
}

#[cfg(test)]
mod tests {
	use super::*;

	fn brute_convolution(f: & Array::<Real, Ix3>, kernel: & Array::<Real, Ix3>, res: &mut Array::<Real, Ix3>){
		assert_eq!(f.shape()[0], res.shape()[0]);
		assert_eq!(f.shape()[1], res.shape()[1]);
		assert_eq!(f.shape()[2], res.shape()[2]);

		assert_eq!(f.shape()[0]*2-1, kernel.shape()[0]);
		assert_eq!(f.shape()[1]*2-1, kernel.shape()[1]);
		assert_eq!(f.shape()[2]*2-1, kernel.shape()[1]);

		for i1 in 0..res.shape()[0]{
			for i2 in 0..res.shape()[0]{
				for i3 in 0..res.shape()[0]{
					res[[i1,i2,i3]] = 0 as Real;
					for j1 in 0..f.shape()[0]{
						for j2 in 0..f.shape()[0]{
							for j3 in 0..f.shape()[0]{
								res[[i1, i2, i3]] += f[[j1,j2,j3]] * kernel[[i1 + f.shape()[0]-1-j1, i2+ f.shape()[1]-1-j2 , i3 + f.shape()[2]-1-j3]];
							}
						}
					}
				}
			}
		}
	}

    #[test]
    fn fft_convolution() {
		let n = 7;
		let n_kernel = 2*n - 1;
		let mut a = Array::zeros((n_kernel, n_kernel, n_kernel));
		let mut b = Array::zeros((n_kernel, n_kernel, n_kernel));
		
		let mut rng = Pcg64::seed_from_u64(0);
		let normal = Normal::new(0 as Real, 1 as Real).unwrap();
		for i1 in 0..n{
			for i2 in 0..n{
				for i3 in 0..n{
					a[[i1,i2,i3]] = normal.sample(&mut rng);
				}
			}
		}
		for i1 in 0..n_kernel{
			for i2 in 0..n_kernel{
				for i3 in 0..n_kernel{
					b[[i1,i2,i3]] = normal.sample(&mut rng);
				}
			}
		}

		let mut a_hat =  Array3::<Complex<Real>>::zeros((n, n_kernel, n_kernel));
		let mut b_hat = Array3::<Complex<Real>>::zeros((n, n_kernel, n_kernel));
		let mut res = Array::zeros((n_kernel, n_kernel, n_kernel));
		let mut res_brute = Array::zeros((n, n, n));
		let mut temp =Array3::<Complex<Real>>::zeros((n, n_kernel, n_kernel));
		fft(&a, &mut a_hat, &mut temp);
		fft(&b, &mut b_hat, &mut temp);
		convolve(&a_hat, &b_hat, &mut res, &mut temp);
		brute_convolution(&a.slice_mut(s![..n, ..n, ..n]).view().to_owned(), &b, &mut res_brute);

		for i1 in 0..n{
			for i2 in 0..n{
				for i3 in 0..n{
					assert_relative_eq!(res[[n-1+i1,n-1+i2,n-1+i3]], res_brute[[i1,i2,i3]], epsilon = (n*n*100) as Real * Real::EPSILON);
				}
			}
		}	
    }
}