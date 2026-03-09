package com.github.kwart.jmicrograd;

import java.util.random.RandomGenerator;

@FunctionalInterface
public interface WeightInit {
    double nextWeight(RandomGenerator rnd, int fanIn);

    public static final WeightInit UNIFORM = (rnd, fanIn) -> rnd.nextDouble(-1.0, 1.0);

    /**
     * Suited for models using sigmoid or tanh activation functions.
     */
    public static final WeightInit LECUN = (rnd, fanIn) -> {
        double limit = Math.sqrt(3.0 / fanIn);
        return rnd.nextDouble(-limit, limit);
    };
    
    /**
     * Suited for ReLU.
     */
    public static final WeightInit KAIMING_NORMAL = (rnd, fanIn) -> {
        double std = Math.sqrt(2.0 / fanIn);
        return rnd.nextGaussian(0.0, std);
    };
}