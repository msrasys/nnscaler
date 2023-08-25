import copy
import logging
import string

from cube.graph.function.dimops import ShapeAnno, OpAnno, IRDimops
from cube.graph.parser.register import CustomizedOps

_logger = logging.getLogger(__name__)


try:
    from apex.normalization.fused_layer_norm import FusedLayerNormFunction, FusedLayerNormAffineFunction, FusedRMSNormFunction, FusedRMSNormAffineFunction
    
    pure_sign_fused_layer_norm = CustomizedOps.create_pure_signature('apex.normalization.fused_layer_norm.FusedLayerNormFunction.apply', True)
    pure_sign_fused_layer_norm_affine = CustomizedOps.create_pure_signature('apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction.apply', True)
    pure_sign_fused_rms_norm = CustomizedOps.create_pure_signature('apex.normalization.fused_layer_norm.FusedRMSNormFunction.apply', True)
    pure_sign_fused_rms_norm_affine = CustomizedOps.create_pure_signature('apex.normalization.fused_layer_norm.FusedRMSNormAffineFunction.apply', True)

    def ApexFusedLayerNormFunction(input, normalized_shape, eps=1e-6, signature = None):
        """
        apex.normalization.fused_layer_norm.FusedLayerNormFunction
        """
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
        return IRDimops(ApexFusedLayerNormFunction, 'fusedlayernorm', signature, [anno], inputs, **kwargs)

    def ApexFusedLayerNormAffineFunction(input, weight, bias, normalized_shape, eps=1e-6, signature = None):
        """
        apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction
        """
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
        return IRDimops(ApexFusedLayerNormAffineFunction, 'fusedlayernormaffine', signature, [anno], inputs, **kwargs)

    def ApexFusedRMSNormFunction(input, normalized_shape, eps=1e-6, signature = None):
        """
        apex.normalization.fused_layer_norm.FusedRMSNormFunction
        """
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
        return IRDimops(ApexFusedRMSNormFunction, 'fusedrmsnorm', signature, [anno], inputs, **kwargs)

    def ApexFusedRMSNormAffineFunction(input, weight, normalized_shape, eps=1e-6, signature = None):
        """
        apex.normalization.fused_layer_norm.FusedRMSNormAffineFunction
        """
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
        anno = OpAnno.create_op_str(einputs, [eoutput])
        kwargs['normalized_shape'] = normalized_shape
        kwargs['eps'] = eps
        return IRDimops(ApexFusedRMSNormAffineFunction, 'fusedrmsnormaffine', signature, [anno], inputs, **kwargs)

    CustomizedOps.register('apex.normalization.fused_layer_norm.FusedLayerNormFunction.apply',
                           ApexFusedLayerNormFunction,
                           f'from apex.normalization.fused_layer_norm import fused_layer_norm as {pure_sign_fused_layer_norm}',
                           FusedLayerNormFunction.apply,
                           keep_full_name=True,
                           trace_autowrap=False)

    CustomizedOps.register('apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction.apply',
                           ApexFusedLayerNormAffineFunction,
                           f'from apex.normalization.fused_layer_norm import fused_layer_norm_affine as {pure_sign_fused_layer_norm_affine}',
                           FusedLayerNormAffineFunction.apply,
                           keep_full_name=True,
                           trace_autowrap=False)

    CustomizedOps.register('apex.normalization.fused_layer_norm.FusedRMSNormFunction.apply',
                           ApexFusedRMSNormFunction,
                           f'from apex.normalization.fused_layer_norm import fused_rms_norm as {pure_sign_fused_rms_norm}',
                           FusedRMSNormFunction.apply,
                           keep_full_name=True,
                           trace_autowrap=False)

    CustomizedOps.register('apex.normalization.fused_layer_norm.FusedRMSNormAffineFunction.apply',
                           ApexFusedRMSNormAffineFunction,
                           f'from apex.normalization.fused_layer_norm import fused_rms_norm_affine as {pure_sign_fused_rms_norm_affine}',
                           FusedRMSNormAffineFunction.apply,
                           keep_full_name=True,
                           trace_autowrap=False)

except:
    _logger.warning('skip apex ops as it is not installed.')
