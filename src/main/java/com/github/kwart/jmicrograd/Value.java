package com.github.kwart.jmicrograd;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import java.util.function.Function;

public class Value {

    private static final Value[] EMPTY_CHILD = new Value[] {};
    private static final Function<Value, Void> BACKPROP_EMPTY = v -> null;

    private final double data;
    private volatile double grad;
    private final Value[] prev;
    private final String op;
    private final Function<Value, Void> backward;

    public Value(double data) {
        this(data, EMPTY_CHILD, BACKPROP_EMPTY, "");
    }

    private Value(double data, Value[] children, Function<Value, Void> backward, String op) {
        this.data = data;
        this.grad = 0.;
        this.prev = children;
        this.backward = backward;
        this.op = op;
    }

    public double getData() {
        return data;
    }

    public double getGrad() {
        return grad;
    }

    public static Value value(double data) {
        return new Value(data);
    }

    private static final Function<Value, Void> BACKPROP_ADD = v -> {
        v.prev[0].grad += v.grad;
        v.prev[1].grad += v.grad;
        return null;
    };

    public Value add(Value other) {
        return new Value(data + other.data, array(this, other), BACKPROP_ADD, "+");
    }

    private static final Function<Value, Void> BACKPROP_MUL = v -> {
        v.prev[0].grad += v.prev[1].data * v.grad;
        v.prev[1].grad += v.prev[0].data * v.grad;
        return null;
    };

    public Value mul(Value other) {
        return new Value(data * other.data, array(this, other), BACKPROP_MUL, "*");
    }

    private static Function<Value, Void> createBackpropFuncPow(double pow) {
        return v -> {
            v.prev[0].grad += (pow * Math.pow(v.prev[0].data, pow - 1.)) * v.grad;
            return null;
        };
    };

    public Value pow(double other) {
        return new Value(Math.pow(this.data, other), array(this), createBackpropFuncPow(other), "*");
    }

    private static final Function<Value, Void> BACKPROP_TANH = v -> {
        v.prev[0].grad += (1. - Math.pow(v.data, 2.)) * v.grad;
        return null;
    };

    public Value tanh() {
        double expp = Math.exp(data);
        double expn = Math.exp(-data);
        return new Value((expp - expn) / (expp + expn), array(this), BACKPROP_TANH, "tanh");
    }

    private static final Function<Value, Void> BACKPROP_RELU = v -> {
        v.prev[0].grad += (v.data > 0 ? v.grad : 0.);
        return null;
    };

    public Value relu() {
        return new Value(data < 0 ? 0. : data, array(this), BACKPROP_RELU, "ReLU");
    }

    private static final Function<Value, Void> BACKPROP_SIGMOID = v -> {
        v.prev[0].grad += (v.data * (1 - v.data)) * v.grad;
        return null;
    };

    public Value sigmoid() {
        double expn = Math.exp(-data);
        return new Value(1. / (1. + expn), array(this), BACKPROP_SIGMOID, "sigmoid");
    }

    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(topo, visited, this);
        this.grad = 1.;
        ListIterator<Value> li = topo.listIterator(topo.size());
        while (li.hasPrevious()) {
            Value v = li.previous();
            v.backward.apply(v);
        }
    }

    private static void buildTopo(List<Value> topo, Set<Value> visited, Value value) {
        if (!visited.contains(value)) {
            visited.add(value);
            for (Value child : value.prev) {
                buildTopo(topo, visited, child);
            }
            topo.add(value);
        }
    }

    private static <T> T[] array(T... values) {
        return values;
    }

    @Override
    public String toString() {
        return String.format("Value [data=%s, grad=%s, op=%s]", data, grad, op);
    }

}
