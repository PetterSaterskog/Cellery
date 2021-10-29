// extern crate cpython;
use pyo3::prelude::*;

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};

mod tumor;
mod fft_convolution;

use tumor::{simulate, DIM, Real};
use crate::tumor::TumorModel;

#[pyfunction]
fn run(py: Python, model: TumorModel, side_length: Real, n_heathy: usize, n_stroma: usize, n_microglia: usize, min_cell_number: usize, thickness: Real) -> (&PyArray2<Real>, &PyArray1<i32>) {
    // let model = read_tumor_model(String::from("ModelFile"));
    let tumor = simulate(model, side_length, n_heathy, n_stroma, n_microglia,min_cell_number);

    //dissect a slice in the center of volume
    let in_slice = |z: Real| -> bool {z >= (side_length-thickness)/2 as Real && z < (side_length+thickness)/2 as Real};
    let n_cells = tumor.cells.iter().filter(|&c| in_slice(c.z)).count();
    let mut cell_positions = Array2::<Real>::zeros((n_cells, DIM));
    let mut cell_types = Array1::<i32>::zeros(n_cells);
    let mut count = 0;
    for c in tumor.cells{
        if in_slice(c.z) {
            cell_positions[[count, 0]] = c.x;
            cell_positions[[count, 1]] = c.y;
            cell_positions[[count, 2]] = c.z;
            cell_types[[count]] = c.cell_type as i32;
            count+=1;
        }
    }
    (numpy::PyArray::from_array(py, &cell_positions), numpy::PyArray::from_array(py, &cell_types))
}

#[pymodule]
fn tumor_model(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TumorModel>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}

