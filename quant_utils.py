# References: 
#   https://arxiv.org/abs/2002.00104
#   https://github.com/jun-fang/PWLQ

import torch
import numpy as np
import torch.nn as nn

##########################################################################################
####  Uniform quantization 
##########################################################################################

def quantize_model_unif(model, nbit, avg_in_layer):
    '''
    Used to quantize the ConvNet model
    '''
    average_qerr = 0
    num_layers = 0
    avg_layer_error = 0
    for m in model.modules():
        avg_layer_error = 0
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            num_layers += 1
            m.weight.data, m.adaptive_scale, err_w_2 = dorefa_g(m.weight, nbit)
            avg_layer_error = err_w_2
            if m.bias is not None:
                m.bias.data, _,_ = dorefa_g(m.bias, nbit, m.adaptive_scale)
            if avg_in_layer:
                avg_layer_error /= (torch.numel(m.weight))
        average_qerr += avg_layer_error
    return average_qerr / num_layers


def quantizer(input, nbit):
    '''
    input: full precision tensor in the range [0, 1]
    return: quantized tensor
    '''
    output = input * (2**nbit -1)
    output = torch.round(output)

    return output/(2**nbit -1)


def dorefa_g(w, nbit, adaptive_scale=None):
    '''
    w: a floating-point weight tensor to quantize
    nbit: the number of bits in the quantized representation
    adaptive_scale: the maximum scale value. if None, it is set to be the
                    absolute maximum value in w.
    '''
    if adaptive_scale is None:
        adaptive_scale = torch.max(torch.abs(w))

    noise_tensor = (torch.rand(w.shape, device=w.device) - 0.5) / (2**nbit - 1)
    intermediate = quantizer(noise_tensor + 0.5 + w / (2*adaptive_scale), nbit)
    w_q = 2 * adaptive_scale * (intermediate - 0.5)

    err = torch.sum(torch.pow(w_q - w, 2))

    return w_q, adaptive_scale, err

##########################################################################################
####  Uniform quantization for PWLQ
##########################################################################################

def uniform_symmetric_quantizer(x, bits=8.0, minv=None, maxv=None, signed=True, 
                                scale_bits=0.0, num_levels=None, scale=None, simulated=True):
    if minv is None:
        maxv = torch.max(torch.abs(x))
        minv = - maxv if signed else 0

    if signed:
        maxv = np.max([-float(minv), float(maxv)])
        minv = - maxv 
    else:
        minv = 0
    
    if num_levels is None:
        num_levels = 2 ** bits

    if scale is None:
        scale = (maxv - minv) / (num_levels - 1)

    if scale_bits > 0:
        scale_levels = 2 ** scale_bits
        scale = torch.round(torch.mul(scale, scale_levels)) / scale_levels
            
    ## clamp
    x = torch.clamp(x, min=float(minv), max=float(maxv))
        
    x_int = torch.round(x / scale)
    
    if signed:
        x_quant = torch.clamp(x_int, min=-num_levels/2, max=num_levels/2 - 1)
        assert(minv == - maxv)
    else:
        x_quant = torch.clamp(x_int, min=0, max=num_levels - 1)
        assert(minv == 0 and maxv > 0)
        
    x_dequant = x_quant * scale
    
    return x_dequant if simulated else x_quant


def uniform_affine_quantizer(x, bits=8.0, minv=None, maxv=None, offset=None, include_zero=False,
                            scale_bits=0.0, num_levels=None, scale=None, simulated=True):
    if minv is None:
        maxv = torch.max(x)
        minv = torch.min(x)
        if include_zero:
            if minv > 0:
                minv = 0
            elif maxv < 0:
                maxv = 0
    
    if num_levels is None:
        num_levels = 2 ** bits
    
    if not scale:
        scale = (maxv - minv) / (num_levels - 1)

    if not offset:
        offset =  minv

    if scale_bits > 0:
        scale_levels = 2 ** scale_bits
        scale = torch.round(torch.mul(scale, scale_levels)) / scale_levels
        offset = torch.round(torch.mul(offset, scale_levels)) / scale_levels
        
    ## clamp
    x = torch.clamp(x, min=float(minv), max=float(maxv))
        
    x_int = torch.round((x - offset) / scale)
    
    x_quant = torch.clamp(x_int, min=0, max=num_levels - 1)
        
    x_dequant = x_quant * scale + offset
    
    return x_dequant if simulated else x_quant


##########################################################################################
####  Piecewise Linear Quantization (PWLQ)
##########################################################################################

#   piecewise linear quantization on range [-m, m] with breakpoint p: 
#       4 pieces: [-m, -p], [-p, 0], [0, p], [p, m]
#   option 1: overlapping 
#       [-p, p], [-m, m] symmetric with b bits signed, zero offsets
#   option 2: non-overlapping
#       [-p, p] symmetric with b bits signed
#       [-m, -p] and [p, m] asymmetric with (b - 1) bits unsigned, non-zero offset p  


def quantize_model_pwlq(model, nbit, avg_in_layer):
    '''
    Used to quantize the ConvNet model wtih PWLQ
    '''
    average_qerr = 0
    num_layers = 0
    avg_layer_error = 0
    for m in model.modules():
        avg_layer_error = 0
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            num_layers += 1
            m.weight.data, m.adaptive_scale, err_w_2 = piecewise_linear_quant(m.weight, bits=nbit, scale_bits=0.0, break_point_approach='norm', pw_opt=2, approximate=False)
            avg_layer_error = err_w_2
            if m.bias is not None:
                m.bias.data, _,err_b_2 = piecewise_linear_quant(m.bias, bits=nbit, scale_bits=0.0, break_point_approach='norm', pw_opt=2, approximate=False)
            if avg_in_layer:
                avg_layer_error /= (torch.numel(m.weight))
        average_qerr += avg_layer_error
    return average_qerr / num_layers


def piecewise_linear_quant(w, bits=4.0, scale_bits=0.0, 
                        break_point_approach='norm', pw_opt=2, approximate=False):
    '''
    Piecewise Linear Quantization (PWLQ)
    '''

    ## assumption: w is satisfying Gaussian or Laplacian distribution
    # break_point_approach = 'norm' (Gaussian) or 'laplace' (Laplacian)
    std_w = torch.std(w) + 1e-12
    abs_max = torch.max(torch.abs(w))
    abs_max_normalized = abs_max / std_w
    break_point_normalized = find_optimal_breakpoint(abs_max_normalized, 
                                pw_opt=pw_opt, dist=break_point_approach, approximate=approximate)
    bkp_ratio = break_point_normalized / abs_max_normalized
    break_point = bkp_ratio * abs_max
    err, qw = pwlq_quant_error(w, bits, scale_bits, abs_max, break_point, pw_opt)
    return qw, bkp_ratio, err


def pwlq_quant_error(w, bits, scale_bits, abs_max, break_point, pw_opt):
    '''
    Piecewise linear quantization (PWLQ) with two options: overlapping or non-overlapping
    '''
    ## option 1: overlapping
    if pw_opt == 1:
        qw_tail = uniform_symmetric_quantizer(w, 
            bits=bits, scale_bits=scale_bits, minv=-abs_max, maxv=abs_max)
        qw_middle = uniform_symmetric_quantizer(w, 
            bits=bits, scale_bits=scale_bits, minv=-break_point, maxv=break_point)
        
        qw = torch.where(-break_point < w, qw_middle, qw_tail)
        qw = torch.where(break_point > w, qw, qw_tail)
    
    ## option 2: non-overlapping
    if pw_opt == 2:
        qw_tail_neg = uniform_affine_quantizer(w, 
            bits=bits-1, scale_bits=scale_bits, minv=-abs_max, maxv=-break_point)
        qw_tail_pos = uniform_affine_quantizer(w, 
            bits=bits-1, scale_bits=scale_bits, minv=break_point, maxv=abs_max)
        qw_middle = uniform_symmetric_quantizer(w, 
            bits=bits, scale_bits=scale_bits, minv=-break_point, maxv=break_point)
    
        qw = torch.where(-break_point < w, qw_middle, qw_tail_neg)
        qw = torch.where(break_point > w, qw, qw_tail_pos)

    err = torch.sum(torch.pow(qw - w, 2))
    return err, qw


def derivative_quant_err(m, p, dist='norm', pw_opt=2):  
    '''
    Compute the derivative of expected variance of quantization error
    '''
    from scipy.stats import norm, laplace
    if dist == 'norm':
        cdf_func = norm.cdf(p)
        pdf_func = norm.pdf(p)
    elif dist == 'laplace':  
        # https://en.wikipedia.org/wiki/Laplace_distribution
        cdf_func = laplace.cdf(p, 0, np.sqrt(0.5))   
        pdf_func = laplace.pdf(p, 0, np.sqrt(0.5)) # pdf(p, a, b) has variance 2*b^2
    else:
        raise RuntimeError("Not implemented for distribution: %s !!!" % dist) 
    
    ## option 1: overlapping
    if pw_opt == 1: 
        # quant_err = [F(p) - F(-p)] * p^2 + 2*[F(m) - F(p)] * m^2
        df_dp = 2 * pdf_func * (p * p - m * m) + 2 * p * (2 * cdf_func - 1.0)
    ## option 2: non-overlapping
    else:  
        # quant_err = [F(p) - F(-p)] * p^2 + 2*[F(m) - F(p)] * (m - p)^2
        df_dp = p - 2 * m + 2 * m * cdf_func + m * pdf_func * (2 * p - m) 

    return df_dp


def gradient_descent(m, pw_opt, dist, lr=0.1, max_iter=100, tol=1e-3):
    '''
    Gradient descent method to find the optimal breakpoint
    '''
    p = m / 2.0
    err, iter_num = 1, 0
    while err > tol and iter_num < max_iter:
        grad = derivative_quant_err(m, p, pw_opt=pw_opt, dist=dist)
        p = p - lr * grad
        err = np.abs(grad)
        iter_num += 1
    return p


def find_optimal_breakpoint(m, pw_opt=2, dist='norm', approximate=False):
    '''
    Find the optimal breakpoint for PWLQ: approximated VS numerical solution
    '''
    m = float(m)
    ## linear approximated solution O(1)
    if approximate:
        assert(pw_opt == 2 and dist in ['norm', 'laplace'])
        if dist.startswith('norm') and pw_opt != 1:
            # Approximated version for Gaussian
            coef = 0.86143114  
            inte = 0.607901097496529 
            break_point = np.log(coef * m + inte)
        elif dist.startswith('laplace') and pw_opt != 1:
            # Approximated version for Laplacian
            coef = 0.80304483
            inte = -0.3166785508381478
            break_point = coef * np.sqrt(m) + inte
    ## numeric solution: gradient descent 
    else:
        # use gradient descent
        break_point = gradient_descent(m, pw_opt, dist)

    ## add random noise
    if 'noise' in dist:
        noise = abs(float(dist.split('-')[-1]))
        rand_num = np.random.uniform(-1.0, 1.0)
        random_noise = noise if rand_num >= 0.0 else -1.0 * noise
        break_point *= (1 + random_noise)
    
    return break_point