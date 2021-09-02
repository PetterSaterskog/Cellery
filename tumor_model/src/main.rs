use rand::prelude::*;
use ndarray::{Array3,Array4, Dim, Ix};
use ndarray::Array;
use ndarray::prelude::*;
use ndrustfft::{ndfft_r2c_par, ndifft_r2c_par, Complex, R2cFftHandler};

type Real = f32;
const DIM: u32 = 3; //not intended to work in other than 3 dimensions, just for code clarity

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

struct TumorModel {
	cell_effective_radii: [Real; 3],
	cancer_growth: Real,
	immune_growth: Real,
	healthy_cancer_diffusion: Real,
	cytotoxicity: Real,
}

struct SimulationAllocation {
	expansion: Array3<Real>,
	displacement: Array4<Real>,
	displacement_kernel_hat: Array4<Real>,
}

struct Tumor {
	model: TumorModel,
	cells: Vec<Cell>,
	size: Real,
	rng: ThreadRng,
}

fn read_tumor_model(_fn: String) -> TumorModel{
	TumorModel{cell_effective_radii:[10.0, 10.0, 10.0],
		cancer_growth:1.0,
		immune_growth:0.1,
		healthy_cancer_diffusion:5.0,
		cytotoxicity:0.5}
}

fn get_initial_tumor(model: TumorModel, size: Real, immune_ratio: Real) -> Tumor {
	let mut vec = Vec::new();
	// let mut vec = Vec::with_capacity(200_000_000);
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
	let min_n = (tumor.size / resolution) as u32;
	//Find first power of 2 large enough (replace with better approx later since we can have more prime factors. This is a huge performance gain since dimensionality is high)
	let mut m=1;
	let n_fft = loop { if m >= 2*min_n - 1 {break m}; m*=2;};
	let n = (n_fft+1)/2; //as large n as possible
	let n_kernel = 2*n-1;

	let coords = Array::zeros((3,4,5,6));

	let mut displacement_kernel = Array::zeros((3,4,5,6));
	for i in 0..DIM {
		displacement_kernel = coords.map_axis(Axis(3), |x| x.powi(2).scalar_sum())
	}
	

	SimulationAllocation {
		expansion: Array::zeros((n, n, n)),
		displacement: Array::zeros((n, n, n, DIM)),
		displacement_kernel_hat:  Array::zeros((n_kernel, n_kernel, n_kernel, DIM)),
	}
}

// #fft of the last d axes
// 	def FFT(self, f):
// 		return fft.rfftn(f, s = self.d*(self.fftSize,))

// #last indices of arguments correspond to spatial grid postions.
// def convolve(self, aFFT, bFFT):
// 	return fft.irfftn( aFFT*bFFT, s = self.d*(self.fftSize,) )[(np.s_[...],) + self.d*(np.s_[self.gridN-1:2*self.gridN-1],)]

fn fft(){ //a: Array::<f32, Ix3>){
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
	println!("inverse..");
	let mut ifft_handler = R2cFftHandler::<Real>::new(nx);
	ndifft_r2c_par(
		&mut vhat.view_mut(),
		&mut data.view_mut(),
		&mut fft_handler,
		0,
	);
	println!("{},{},{},{}", data_c[[0,0,0]], data_c[[0,0,1]], data_c[[0,0,2]], data_c[[0,0,3]]);
	println!("{},{},{},{}", data[[0,0,0]], data[[0,0,1]], data[[0,0,2]], data[[0,0,3]]);
}

// fn convolve(a_fft: Array::<Real, Ix3>, b_fft: Array::<Real, Ix3>)
// {
// 	ndifft_r2c_par

// }

fn update_tumor(tumor: &mut Tumor, dt: Real){
	//add cell ids in spatial grid
	//create index in parallel
	for i in 0..tumor.cells.len(){

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
	let mut tumor = get_initial_tumor(model, 1000.0, 0.5);
	let resolution = 12.;
	let n_threads = 4;

	println!("Initializing simulation..");
	let all = initialize_simulation(tumor, resolution, n_threads);


	let dt = 0.1;
	loop {
		
		println!("Number of cells: {}", tumor.cells.len());
		if tumor.cells.len() >= max_tumor_cells{
			break;
		}
		update_tumor(&mut tumor, dt);
		
	}
	tumor
}

fn main() {
	fft();

	let model = read_tumor_model(String::from("ModelFile"));
	simulate(model, 1_000_000);
}