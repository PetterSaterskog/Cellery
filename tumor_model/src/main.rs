use rand::prelude::*;
use ndarray::{Array3,Array4};
use ndarray::Array;
use ndarray::prelude::*;
use ndrustfft::{ndfft_r2c_par, ndifft_r2c_par, Complex, R2cFftHandler};

type Real = f32;
const DIM: usize = 3; //not intended to work in other than 3 dimensions, just for code clarity

#[derive(Clone)]
enum CellType {
	Healthy,
	Cancer,
	Immune,
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
	displacement: Array4<Real>,
	displacement_kernel_hat: Array4<Complex<Real>>,
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

	let mut displacement_kernel = Array::zeros((3,4,5,6));
	
	displacement_kernel = kernel_coords.clone();
	
	let mut displacement_kernel_hat = Array4::<Complex<Real>>::zeros((n_kernel, n_kernel, n_kernel, DIM));
	
	for i in 0..DIM
	{
		fft(displacement_kernel.index_axis_mut(Axis(DIM), i).to_owned(), displacement_kernel_hat.index_axis_mut(Axis(DIM), i).to_owned());
	}
	
	let index_n = tumor.size / tumor.model.neighbor_distance;

	SimulationAllocation {
		expansion: Array::zeros((n, n, n)),
		displacement: Array::zeros((n, n, n, DIM)),
		displacement_kernel_hat: displacement_kernel_hat,
		indexN: Array::zeros((index_n, index_n, index_n)),
		indexN: Array::zeros((index_n, index_n, index_n)),
	}
}

// #fft of the last d axes
// 	def FFT(self, f):
// 		return fft.rfftn(f, s = self.d*(self.fftSize,))

// #last indices of arguments correspond to spatial grid postions.
// def convolve(self, aFFT, bFFT):
// 	return fft.irfftn( aFFT*bFFT, s = self.d*(self.fftSize,) )[(np.s_[...],) + self.d*(np.s_[self.gridN-1:2*self.gridN-1],)]

fn fft(a:Array3::<Real>, a_hat:Array3::<Complex<Real>>){ //a: Array::<f32, Ix3>){
	println!("allocating..");
	let (nx, ny, nz) = (400, 400, 400);
	let mut data = Array3::<Real>::zeros((nx, ny, nz));
	let mut vhat = Array3::<Complex<Real>>::zeros((nx / 2 + 1, ny, nz));
	for (i, v) in data.iter_mut().enumerate() {
		*v = (i%23) as Real;
	}
	let data_c = data.clone();
	
	let mut fft_handler = R2cFftHandler::<Real>::new(nx);

	println!("transform..");
	ndfft_r2c_par(
		&mut data.view_mut(),
		&mut vhat.view_mut(),
		&mut fft_handler,
		0,
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

// fn convolve(a_fft: Array::<Real, Ix3>, b_fft: Array::<Real, Ix3>)
// {
// 	println!("inverse..");
// 	let mut ifft_handler = R2cFftHandler::<Real>::new(nx);
// 	ndifft_r2c_par(
// 		&mut vhat.view_mut(),
// 		&mut data.view_mut(),
// 		&mut ifft_handler,
// 		0,
// 	);
// }


fn update_tumor(tumor: &mut Tumor, all: &mut SimulationAllocation, dt: Real){
	//add cell ids in spatial grid
	//create index in parallel
	
	all.index_n.fill(0)
	for i in 0..tumor.cells.len(){
		let mut corner_indices: Vec<[usize, DIM]>; 
		let mut corner = [bool, DIM];
		let (px, py, pz) = (tumor.cells[i].x, tumor.cells[i].y, tumor.cells[i].z)
		for ci in 0..2{
			let gi = (px + )
		}
		loop {
			corner
			loop{

			}
		}
	}


	//diffuse cells in parallel

	//split some cells. Add to displacement histogram in parallel
	for i in 0..tumor.cells.len() {
		match tumor.cells[i].cell_type {
			CellType::Cancer  => {
				if true || tumor.rng.gen::<Real>() < tumor.model.cancer_growth*dt {
					tumor.cells.push(tumor.cells[i].clone())
				}
			}
			CellType::Immune  => {
				if tumor.rng.gen::<Real>() < tumor.model.immune_growth*dt {
					tumor.cells.push(tumor.cells[i].clone())
				}
			}
			_ => (),
		}
	}
}

fn save_tumor(tumor: &Tumor){
	
}

fn simulate(model:TumorModel, max_tumor_cells: usize) -> Tumor{
	let mut tumor = get_initial_tumor(model, 50.0, 0.5);
	let resolution = 12.;
	let n_threads = 4;

	println!("Initializing simulation..");
	let allocation = initialize_simulation(&tumor, resolution, n_threads);


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
