module nf_loss

   !! This module provides a collection of loss functions and their derivatives.
   !! The implementation is based on an abstract loss derived type
   !! which has the required eval and derivative methods.
   !! An implementation of a new loss type thus requires writing a concrete
   !! loss type that extends the abstract loss derived type, and that
   !! implements concrete eval and derivative methods that accept vectors.

   implicit none

   private
   public :: loss_type
   public :: mse
   public :: quadratic
   public :: binary_cross_entropy

   type, abstract :: loss_type
   contains
      procedure(loss_interface), nopass, deferred :: eval
      procedure(loss_derivative_interface), nopass, deferred :: derivative
   end type loss_type

   abstract interface
      pure function loss_interface(true, predicted) result(res)
         real, intent(in) :: true(:)
         real, intent(in) :: predicted(:)
         real :: res
      end function loss_interface
      pure function loss_derivative_interface(true, predicted) result(res)
         real, intent(in) :: true(:)
         real, intent(in) :: predicted(:)
         real :: res(size(true))
      end function loss_derivative_interface
   end interface

   type, extends(loss_type) :: mse
      !! Mean Square Error loss function
   contains
      procedure, nopass :: eval => mse_eval
      procedure, nopass :: derivative => mse_derivative
   end type mse

   type, extends(loss_type) :: quadratic
      !! Quadratic loss function
   contains
      procedure, nopass :: eval => quadratic_eval
      procedure, nopass :: derivative => quadratic_derivative
   end type quadratic

   type, extends(loss_type) :: binary_cross_entropy
      !! Binary Cross-Entropy loss function
   contains
      procedure, nopass :: eval => binary_cross_entropy_eval
      procedure, nopass :: derivative => binary_cross_entropy_derivative
   end type binary_cross_entropy

   interface

      pure module function mse_eval(true, predicted) result(res)
         !! Mean Square Error loss function:
         !!
         !!   L  = sum((predicted - true)**2) / size(true)
         !!
         real, intent(in) :: true(:)
         !! True values, i.e. labels from training datasets
         real, intent(in) :: predicted(:)
         !! Values predicted by the network
         real :: res
         !! Resulting loss value
      end function mse_eval

      pure module function mse_derivative(true, predicted) result(res)
         !! First derivative of the Mean Square Error loss function:
         !!
         !!   L  = 2 * (predicted - true) / size(true)
         !!
         real, intent(in) :: true(:)
         !! True values, i.e. labels from training datasets
         real, intent(in) :: predicted(:)
         !! Values predicted by the network
         real :: res(size(true))
         !! Resulting loss values
      end function mse_derivative

      pure module function quadratic_eval(true, predicted) result(res)
         !! Quadratic loss function:
         !!
         !!   L  = sum((predicted - true)**2) / 2
         !!
         real, intent(in) :: true(:)
         !! True values, i.e. labels from training datasets
         real, intent(in) :: predicted(:)
         !! Values predicted by the network
         real :: res
         !! Resulting loss value
      end function quadratic_eval

      pure module function quadratic_derivative(true, predicted) result(res)
         !! First derivative of the quadratic loss function:
         !!
         !!   L' =  predicted - true
         !!
         real, intent(in) :: true(:)
         !! True values, i.e. labels from training datasets
         real, intent(in) :: predicted(:)
         !! Values predicted by the network
         real :: res(size(true))
         !! Resulting loss values
      end function quadratic_derivative

      pure module function binary_cross_entropy_eval(true, predicted) result(res)
         !! Binary Cross-Entropy loss function:
         !!
         !!   given a sigmoid activation function, the binary cross-entropy loss is defined as:
         !!   L  = - sum(true * log(predicted)) - sum((1 - true) * log(1 - predicted))
         !!
         !!   given a linear activation function, the binary cross-entropy loss is defined as:
         !!   L = sum((1 - true) * predicted + log(1 + exp(-predicted)))
         !!
         real, intent(in) :: true(:)
         !! True values, i.e. labels from training datasets
         real, intent(in) :: predicted(:)
         !! Values predicted by the network
         real :: res
         !! Resulting loss value
      end function binary_cross_entropy_eval

      pure module function binary_cross_entropy_derivative(true, predicted) result(res)
         !! First derivative of the binary cross-entropy loss function:
         !!
         !!   given a sigmoid activation function, the derivative of the binary cross-entropy loss is defined as:
         !!   L' =  - true / predicted + (1 - true) / (1 - predicted)
         !!
         !!   given a linear activation function, the derivative of the binary cross-entropy loss is defined as:
         !!   L' =  (1 - true) - 1 / (1 + exp(predicted))
         !!
         real, intent(in) :: true(:)
         !! True values, i.e. labels from training datasets
         real, intent(in) :: predicted(:)
         !! Values predicted by the network
         real :: res(size(true))
         !! Resulting loss values
      end function binary_cross_entropy_derivative

   end interface

end module nf_loss
