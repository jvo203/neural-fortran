submodule(nf_layer_constructors) nf_layer_constructors_submodule

  use nf_layer, only: layer
  use nf_conv2d_layer, only: conv2d_layer
  use nf_dense_layer, only: dense_layer
  use nf_input1d_layer, only: input1d_layer
  use nf_input3d_layer, only: input3d_layer
  use nf_maxpool2d_layer, only: maxpool2d_layer

  implicit none

contains

  pure module function input1d(layer_size) result(res)
    integer, intent(in) :: layer_size
    type(layer) :: res
    res % name = 'input'
    res % layer_shape = [layer_size]
    res % input_layer_shape = [integer ::]
    allocate(res % p, source=input1d_layer(layer_size))
    res % initialized = .true.
  end function input1d


  pure module function input3d(layer_shape) result(res)
    integer, intent(in) :: layer_shape(3)
    type(layer) :: res
    res % name = 'input'
    res % layer_shape = layer_shape
    res % input_layer_shape = [integer ::]
    allocate(res % p, source=input3d_layer(layer_shape))
    res % initialized = .true.
  end function input3d


  pure module function dense(layer_size, activation) result(res)
    integer, intent(in) :: layer_size
    character(*), intent(in), optional :: activation
    type(layer) :: res

    res % name = 'dense'
    res % layer_shape = [layer_size]

    if (present(activation)) then
      res % activation = activation
    else
      res % activation = 'sigmoid'
    end if

    allocate(res % p, source=dense_layer(layer_size, res % activation))

  end function dense


  pure module function conv2d(window_size, filters, activation) result(res)
    integer, intent(in) :: window_size
    integer, intent(in) :: filters
    character(*), intent(in), optional :: activation
    type(layer) :: res

    res % name = 'conv2d'

    if (present(activation)) then
      res % activation = activation
    else
      res % activation = 'sigmoid'
    end if

    allocate( &
      res % p, &
      source=conv2d_layer(window_size, filters, res % activation) &
    )

  end function conv2d


  pure module function maxpool2d(pool_size, stride) result(res)
    integer, intent(in) :: pool_size
    integer, intent(in), optional :: stride
    integer :: stride_
    type(layer) :: res

    res % name = 'maxpool2d'

    if (present(stride)) then
      stride_ = stride
    else
      stride_ = pool_size
    end if

    allocate( &
      res % p, &
      source=maxpool2d_layer(pool_size, stride) &
    )

  end function maxpool2d

end submodule nf_layer_constructors_submodule
