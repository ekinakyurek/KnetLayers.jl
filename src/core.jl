# Main design principles:
# Layer constists of callable objects composition of primitives, rnns , cnns etc
# Model consists of Layers
# TODO :
#   - design Loss functionality
#   - design optimizer functionality


abstract type Layer end;

abstract type Loss <: Layer end;

abstract type Activation <: Layer end;
