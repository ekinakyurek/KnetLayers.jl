# TODO :
#   - design optimizer functionality


abstract type Layer end

abstract type Loss <: Layer end

abstract type Activation <: Layer end

abstract type AbstractRNN <: Layer end
