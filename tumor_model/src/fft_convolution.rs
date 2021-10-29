use approx::assert_relative_eq;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::{Array3, Array4};
use ndrustfft::{
    ndfft_par, ndfft_r2c_par, ndifft_par, ndifft_r2c_par, Complex, FftHandler, R2cFftHandler,
};
use rand_distr::{Distribution, Normal, UnitSphere};

type Real = f32;

pub fn fft(a: &Array3<Real>, a_hat: &mut Array3<Complex<Real>>, temp: &mut Array3<Complex<Real>>) {
    //a: Array::<f32, Ix3>){
    let nx = a.shape()[0];
    let mut rfft_handler = R2cFftHandler::<Real>::new(nx);
    let mut fft_handler = FftHandler::<Real>::new(nx);

    ndfft_r2c_par(a, a_hat, &mut rfft_handler, 0);
    ndfft_par(a_hat, temp, &mut fft_handler, 1);
    ndfft_par(temp, a_hat, &mut fft_handler, 2);
}

// Convolve a * b -> c
pub fn convolve(
    a_hat: &Array<Complex<Real>, Ix3>,
    b_hat: &Array<Complex<Real>, Ix3>,
    c: &mut Array<Real, Ix3>,
    temp: &mut Array<Complex<Real>, Ix3>,
) {
    let nx = a_hat.shape()[0] * 2 - 1;
    let ny = a_hat.shape()[1];
    let nz = a_hat.shape()[2];
    assert_eq!(nx, ny);
    assert_eq!(nx, nz);

    let mut ab_hat = a_hat * b_hat;
    let mut irfft_handler = R2cFftHandler::<Real>::new(nx);
    let mut ifft_handler = FftHandler::<Real>::new(nx);

    ndifft_par(&ab_hat, temp, &mut ifft_handler, 2);
    ndifft_par(temp, &mut ab_hat, &mut ifft_handler, 1);
    ndifft_r2c_par(&ab_hat, c, &mut irfft_handler, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    type Real = f32;

    use rand_pcg::Pcg64;

    fn brute_convolution(
        f: &Array<Real, Ix3>,
        kernel: &Array<Real, Ix3>,
        res: &mut Array<Real, Ix3>,
    ) {
        assert_eq!(f.shape()[0], res.shape()[0]);
        assert_eq!(f.shape()[1], res.shape()[1]);
        assert_eq!(f.shape()[2], res.shape()[2]);
        assert_eq!(f.shape()[0] * 2 - 1, kernel.shape()[0]);
        assert_eq!(f.shape()[1] * 2 - 1, kernel.shape()[1]);
        assert_eq!(f.shape()[2] * 2 - 1, kernel.shape()[1]);

        for i1 in 0..res.shape()[0] {
            for i2 in 0..res.shape()[0] {
                for i3 in 0..res.shape()[0] {
                    res[[i1, i2, i3]] = 0 as Real;
                    for j1 in 0..f.shape()[0] {
                        for j2 in 0..f.shape()[0] {
                            for j3 in 0..f.shape()[0] {
                                res[[i1, i2, i3]] += f[[j1, j2, j3]]
                                    * kernel[[
                                    (i1 + kernel.shape()[0] - j1) % kernel.shape()[0],
                                    (i2 + kernel.shape()[1] - j2) % kernel.shape()[1],
                                    (i3 + kernel.shape()[2] - j3) % kernel.shape()[2],
                                    ]];
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
        let n_kernel = 2 * n - 1;
        let mut a = Array::zeros((n_kernel, n_kernel, n_kernel));
        let mut b = Array::zeros((n_kernel, n_kernel, n_kernel));

        let mut rng = Pcg64::seed_from_u64(0);
        let normal = Normal::new(0 as Real, 1 as Real).unwrap();
        for i1 in 0..n {
            for i2 in 0..n {
                for i3 in 0..n {
                    a[[i1, i2, i3]] = normal.sample(&mut rng);
                }
            }
        }
        for i1 in 0..n_kernel {
            for i2 in 0..n_kernel {
                for i3 in 0..n_kernel {
                    b[[i1, i2, i3]] = normal.sample(&mut rng);
                }
            }
        }

        let mut a_hat = Array3::<Complex<Real>>::zeros((n, n_kernel, n_kernel));
        let mut b_hat = Array3::<Complex<Real>>::zeros((n, n_kernel, n_kernel));
        let mut res = Array::zeros((n_kernel, n_kernel, n_kernel));
        let mut res_brute = Array::zeros((n, n, n));
        let mut temp = Array3::<Complex<Real>>::zeros((n, n_kernel, n_kernel));
        fft(&a, &mut a_hat, &mut temp);
        fft(&b, &mut b_hat, &mut temp);
        convolve(&a_hat, &b_hat, &mut res, &mut temp);
        brute_convolution(
            &a.slice_mut(s![..n, ..n, ..n]).view().to_owned(),
            &b,
            &mut res_brute,
        );

        for i1 in 0..n {
            for i2 in 0..n {
                for i3 in 0..n {
                    assert_relative_eq!(
                        //res[[n - 1 + i1, n - 1 + i2, n - 1 + i3]],
                        res[[i1,i2,i3]],
                        res_brute[[i1, i2, i3]],
                        epsilon = (n * n * 100) as Real * Real::EPSILON
                    );
                }
            }
        }
    }
}
