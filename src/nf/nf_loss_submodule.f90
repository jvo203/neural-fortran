submodule(nf_loss) nf_loss_submodule

   implicit none

contains

   pure module function quadratic_eval(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res
      res = sum((predicted - true)**2) / 2
   end function quadratic_eval

   pure module function quadratic_derivative(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res(size(true))
      res = predicted - true
   end function quadratic_derivative

   pure module function mse_eval(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res
      res = sum((predicted - true)**2) / size(true)
   end function mse_eval

   pure module function mse_derivative(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res(size(true))
      res = 2 * (predicted - true) / size(true)
   end function mse_derivative

   pure module function binary_cross_entropy_eval(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res
      res = -sum(true * log(predicted) + (1 - true) * log(1 - predicted))
   end function binary_cross_entropy_eval

   pure module function binary_cross_entropy_derivative(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res(size(true))
      res = -true / predicted + (1 - true) / (1 - predicted)
   end function binary_cross_entropy_derivative

end submodule nf_loss_submodule
