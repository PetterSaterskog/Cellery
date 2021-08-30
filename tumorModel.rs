enum CellType {
	Healthy,
	Cancer,
	Immune,
}

struct Cell {
	x: f32,
	y: f32,
	cell_type: CellType,
}

struct Tumor {
	cells: Vec<Cell>,
	size: f32,
}

fn get_initial_tumor(size: f32) -> Tumor {
	let mut vec = Vec::new();
	let c = Cell {x:size/2., y:size/2., cell_type:CellType::Cancer};
	vec.push(c);
	Tumor{size:size, cells:vec}
}

fn update_tumor(tumor: &mut Tumor){
	for _c in &tumor.cells{
		println!("split");
	}
	tumor.cells.push(Cell {x:tumor.size/2., y:tumor.size/2., cell_type:CellType::Cancer})
}

fn save_tumor(tumor: &Tumor){
	for _c in &tumor.cells{
		println!("split");
	}
	tumor.cells.push(Cell {x:tumor.size/2., y:tumor.size/2., cell_type:CellType::Cancer})
}

fn simulate(maxTumorCells: usize) -> Tumor{
	let mut tumor = get_initial_tumor(1000.0);

	println!("Number of cells: {}", tumor.cells.len());

	loop {
		println!("Number of cells: {}", tumor.cells.len());
		if tumor.cells.len() >= maxTumorCells{
			break;
		}
		update_tumor(&mut tumor)
	}
	tumor
}

fn main() {
	simulate(100);
}