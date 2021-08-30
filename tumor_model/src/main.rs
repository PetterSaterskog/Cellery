enum CellType {
	Healthy,
	Cancer,
	Immune,
}

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

struct Tumor {
	model: TumorModel,
	cells: Vec<Cell>,
	size: f32,
}

fn read_tumor_model(_fn: String) -> TumorModel{
	TumorModel{cell_effective_radii:[10.0, 10.0, 10.0] ,cancer_growth:1.0,immune_growth:0.1,healthy_cancer_diffusion:5.0,cytotoxicity:0.5}
}

fn get_initial_tumor(model: TumorModel, size: f32, immuneRatio: f32) -> Tumor {
	let mut vec = Vec::new();
	let half = size/2.0;
	let nHealthy = 10;
	for i in 0..nHealthy{
		let c = Cell {x:half, y:half, z:half, cell_type:CellType::Healthy};
		vec.push(c);
	}
	let c = Cell {x:half, y:half, z:half, cell_type:CellType::Cancer};
	vec.push(c);
	Tumor{model, size:size, cells:vec}
}

fn update_tumor(tumor: &mut Tumor){
	for _c in &tumor.cells{
		println!("split");
	}
	tumor.cells.push(Cell {x:tumor.size/2., y:tumor.size/2., z:tumor.size/2., cell_type:CellType::Cancer})
}

fn save_tumor(tumor: &mut Tumor){
	for _c in &tumor.cells{
		println!("split");
	}
	tumor.cells.push(Cell {x:tumor.size/2., y:tumor.size/2., z:tumor.size/2., cell_type:CellType::Cancer})
}

fn simulate(model:TumorModel, max_tumor_cells: usize) -> Tumor{
	let mut tumor = get_initial_tumor(model, 1000.0, 0.5);

	println!("Number of cells: {}", tumor.cells.len());

	loop {
		println!("Number of cells: {}", tumor.cells.len());
		if tumor.cells.len() >= max_tumor_cells{
			break;
		}
		update_tumor(&mut tumor)
	}
	tumor
}

fn main() {
	let model = read_tumor_model(String::from("ModelFile"));
	simulate(model, 100);
}