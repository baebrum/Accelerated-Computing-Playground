Use OpenMP parallel for and also parallel reduction to speed up the composite ⅓ Simpson’s quadrature method.

An example of an integral that is challenging to solve analytically is the following: $\int_0^{\frac{\pi}{2}}\arccos\left(\frac{\cos\left(x\right)}{1+2\cos\left(x\right)}\right)dx$

It can be shown that this integral has an exact solution of $\frac{5\pi^2}{24}\frac{}{}$

Your code should display the numerical error, i.e., the absolute value of the difference between the approximated integral value and the exact solution, and verify this error is acceptable, say in the order of $10^{-14}$ or smaller.

Execute your parallel implementation on [dgx.sdsu.edu](http://dgx.sdsu.edu/) with a hardware thread count as a power of 2 from $2^{0}$ thread to $2^{8}$ threads, and generate a runtime plot.

Plot runtime in units of seconds as a function of thread count. At what particular thread count do you find the runtime to be minimum or optimal?