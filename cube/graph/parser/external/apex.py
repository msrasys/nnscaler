import copy
import logging
import string

from cube.graph.function.dimops import ShapeAnno, OpAnno, IRDimops
from cube.graph.parser.register import CustomizedOps

_logger = logging.getLogger(__name__)


try:
    from apex.normalization.fused_layer_norm import FusedLayerNormFunction, FusedLayerNormAffineFunction

    def ApexFusedLayerNormFunction(input, normalized_shape, eps=1e-6, signature = None):
        """
        apex.normalization.fused_layer_norm.FusedLayerNormFunction
        """
        signature = 'apex_fused_layer_norm'
        letters = iter(string.ascii_lowercase)
        einput = ShapeAnno.create_shape_str(input.shape, iterator=letters)
        eoutput = copy.copy(einput)
        ndims = len(input.shape)
        for dim in range(len(normalized_shape)):
            einput[ndims-1-dim] += '^'
            eoutput[ndims-1-dim] += '^'
        einputs, inputs = [einput], [input]
        kwargs = {}
        anno = OpAnno.create_op_str(einputs, [eoutput])
        kwargs['normalized_shape'] = normalized_shape
        kwargs['eps'] = eps
        return IRDimops(FusedLayerNormFunction, 'fusedlayernorm', signature, [anno], inputs, **kwargs)

    def ApexFusedLayerNormAffineFunction(input, weight, bias, normalized_shape, eps=1e-6, signature = None):
        """
        apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction
        """
        signature = 'apex_fused_layer_norm_affine'
        assert not (weight is None and bias is not None), f"Not support for None of weight and parameter of bias"
        letters = iter(string.ascii_lowercase)
        einput = ShapeAnno.create_shape_str(input.shape, iterator=letters)
        eoutput = copy.copy(einput)
        ndims = len(input.shape)
        for dim in range(len(normalized_shape)):
            einput[ndims-1-dim] += '^'
            eoutput[ndims-1-dim] += '^'
        einputs, inputs = [einput], [input]
        kwargs = {}
        if weight is not None:
            eweight = ShapeAnno.create_shape_str(weight.shape, reduction='^', iterator=letters)
            einputs.append(eweight)
            inputs.append(weight)
        else:
            kwargs['weight'] = weight
        if bias is not None:
            ebias = ShapeAnno.create_shape_str(bias.shape, reduction='^', iterator=letters)
            einputs.append(ebias)
            inputs.append(bias)
        else:
            kwargs['bias'] = bias
        anno = OpAnno.create_op_str(einputs, [eoutput])
        kwargs['normalized_shape'] = normalized_shape
        kwargs['eps'] = eps
        return IRDimops(FusedLayerNormAffineFunction, 'fusedlayernormaffine', signature, [anno], inputs, **kwargs)
    
    CustomizedOps.register('apex.normalization.fused_layer_norm.FusedLayerNormFunction.apply',
                           ApexFusedLayerNormFunction,
                           'from apex.normalization.fused_layer_norm import fused_layer_norm as apex_fused_layer_norm',
                           FusedLayerNormFunction.apply,
                           keep_full_name=True,
                           trace_autowrap=False)

    CustomizedOps.register('apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction.apply',
                           ApexFusedLayerNormAffineFunction,
                           'from apex.normalization.fused_layer_norm import fused_layer_norm_affine as apex_fused_layer_norm_affine',
                           FusedLayerNormAffineFunction.apply,
                           keep_full_name=True,
                           trace_autowrap=False)
except:
    _logger.warning('skip apex ops as it is not installed.')


