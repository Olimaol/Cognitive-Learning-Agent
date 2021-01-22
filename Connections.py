#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:28:47 2018

@author: alexschw
"""
import ANNarchy as ann
from functions import rangeX


def one_to_dim(pre, post, postDim, weight=1.0, delays=0):
    """
    Creates a connection that connects each neuron of the pre-population to
    all neurons along the specified post-dimension (postDim).
    """
    assert pre.geometry[0] == post.geometry[postDim]

    synapses = ann.CSR()
    nGeometry = post.geometry[:postDim] + (1,) + post.geometry[postDim + 1:]
    for m in rangeX(nGeometry):
        for n in range(pre.geometry[0]):
            mn = m[:postDim] + (n,) + m[postDim + 1:]
            synapses.add(post.rank_from_coordinates(mn), [n],
                         [weight], [delays])
    return synapses
"""
def one_to_dim_gauss(pre, post, postDim, weight=1.0, delays=0):
    assert pre.geometry[0] == post.geometry[postDim]

    synapses = ann.CSR()
    nGeometry = post.geometry[:postDim] + (1,) + post.geometry[postDim+1:]
    
    for m in rangeX(nGeometry):
        for pre_n in xrange(pre.geometry[0]):
		for post_n in xrange(pre.geometry[0]):
		    mn = m[:postDim]+(post_n,)+m[postDim+1:]
		    print pre_n, mn
		    synapses.add(post.rank_from_coordinates(mn), [pre_n], [weight], [delays])
    return synapses
"""


def dim_to_one(pre, post, preDim, weight=1.0, delays=0):
    """
    Creates a connection that connects all neurons along the specified
    pre-dimension (preDim) to one neuron of the pre-population.
    """
    assert post.geometry[0] == pre.geometry[preDim]

    synapses = ann.CSR()
    nGeometry = pre.geometry[:preDim] + (1,) + pre.geometry[preDim + 1:]
    for n in range(post.geometry[0]):
        pre_ranks = []
        for m in rangeX(nGeometry):
            mn = m[:preDim] + (n,) + m[preDim + 1:]
            pre_ranks.append(pre.rank_from_coordinates(mn))
        synapses.add(n, pre_ranks, [weight], [delays])
    return synapses


def con_scale(pre, post, factor, weight=1.0, delays=0):
    """
    Lineary scales a projection e.g. to connect a 2x2 with a 4x4 population.
    The factor is indicating how much the neurons are scaled in any dimension
    """
    synapses = ann.CSR()

    for n in range(post.geometry[0]):
        for m in range(post.geometry[1]):
            post_rank = post.rank_from_coordinates((n, m))
            pre_rank = pre.rank_from_coordinates((n // factor, m // factor))
            synapses.add(post_rank, [pre_rank], [weight], [delays])
    return synapses


def mlayer_to_layer(pre, post, layer, w=1.0, delays=0):
    """
    Connects a multidimensional population to a a post projection where the
    layer given equals to one but does not differ otherwise. Performs a sum-
    pooling operation when w is equal to 1.
    """
    target_post_geom = (pre.geometry[:layer] + (1,) + pre.geometry[layer + 1:])
    assert target_post_geom == post.geometry

    synapses = ann.CSR()
    for n in rangeX(post.geometry):
        pre_ranks = []
        for m in range(pre.geometry[layer]):
            pre_loc = pre.geometry[:layer] + (m,) + pre.geometry[layer + 1:]
            pre_ranks.append(pre.rank_from_coordinates(pre_loc))
        synapses.add(n, pre_ranks, [w], [delays])
    return synapses


def gauss_to_layer(pre, post, preLayer, amp=1.0, sigma=0.5):
    """
    Implements the connect_to_gaussian for multiple layers
    """
    synapses = ann.CSR()
    for n in rangeX(post.geometry):
        pass
    return synapses

