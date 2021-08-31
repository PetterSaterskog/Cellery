use rand::prelude::*;

#[derive(Clone)]
enum CellType {
	Healthy,
	Cancer,
	Immune,
}

#[derive(Clone)]
struct Cell {
	x: f32,
	y: f32,
	z: f32,
	cell_type: CellType,
}

struct TumorModel {
	cell_effective_radii: [f32; 3],
	cancer_growth: f32,
	immune_growth: f32,
	healthy_cancer_diffusion: f32,
	cytotoxicity: f32,
}

struct SpatialIndex {
	
}

struct Tumor {
	model: TumorModel,
	cells: Vec<Cell>,
	size: f32,
	rng: ThreadRng,
}

fn read_tumor_model(_fn: String) -> TumorModel{
	TumorModel{cell_effective_radii:[10.0, 10.0, 10.0], cancer_growth:1.0, immune_growth:0.1, healthy_cancer_diffusion:5.0, cytotoxicity:0.5}
}

fn get_initial_tumor(model: TumorModel, size: f32, immune_ratio: f32) -> Tumor {
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

fn update_tumor(tumor: &mut Tumor, dt: f32){
	//add cell ids in spatial grid
	//create index in parallel
	for i in 0..tumor.cells.len(){

	}

	//diffuse cells in parallel

	//split some cells. Add to displacement histogram in parallel
	for i in 0..tumor.cells.len() {
		match tumor.cells[i].cell_type {
			CellType::Cancer  => {
				if true || tumor.rng.gen::<f32>() < tumor.model.cancer_growth*dt {
					tumor.cells.push(tumor.cells[i].clone())
				}
			}
			CellType::Immune  => {
				if tumor.rng.gen::<f32>() < tumor.model.immune_growth*dt {
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
	let dt = 0.1;
	loop {
		println!("Number of cells: {}", tumor.cells.len());
		if tumor.cells.len() >= max_tumor_cells{
			break;
		}
		update_tumor(&mut tumor, dt)
	}
	tumor
}

fn main() {
	let model = read_tumor_model(String::from("ModelFile"));
	simulate(model, 100_000_000);
}