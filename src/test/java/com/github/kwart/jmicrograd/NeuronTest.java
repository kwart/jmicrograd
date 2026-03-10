package com.github.kwart.jmicrograd;

import static com.github.kwart.jmicrograd.Value.value;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class NeuronTest {

    private static final double TOL = 1e-6;

    @Test
    void constructorRejectsZeroInputs() {
        assertThatThrownBy(() -> new Neuron(0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void constructorRejectsNegativeInputs() {
        assertThatThrownBy(() -> new Neuron(-1))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void parametersLengthIsInputsPlusOne() {
        Neuron n = new Neuron(3);
        assertThat(n.parametersLength()).isEqualTo(4);
        assertThat(n.parameters()).hasSize(4);
    }

    @Test
    void parametersLengthSingleInput() {
        Neuron n = new Neuron(1);
        assertThat(n.parametersLength()).isEqualTo(2);
        assertThat(n.parameters()).hasSize(2);
    }

    @Test
    void applyReturnsTanhActivation() {
        Neuron n = new Neuron(2);
        Value[] x = { value(1.0), value(2.0) };
        Value out = n.apply(x);

        // Output should be tanh of the weighted sum, so in [-1, 1]
        assertThat(out.getData()).isBetween(-1.0, 1.0);
    }

    @Test
    void applyRejectsNullInput() {
        Neuron n = new Neuron(2);
        assertThatThrownBy(() -> n.apply(null))
                .isInstanceOf(NullPointerException.class);
    }

    @Test
    void applyRejectsWrongInputLength() {
        Neuron n = new Neuron(3);
        Value[] x = { value(1.0), value(2.0) };
        assertThatThrownBy(() -> n.apply(x))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void applySupportsBackpropagation() {
        Neuron n = new Neuron(2);
        Value[] x = { value(1.0), value(2.0) };
        Value out = n.apply(x);
        out.backward();

        // After backward, input gradients should be set
        assertThat(x[0].getGrad()).isNotNaN();
        assertThat(x[1].getGrad()).isNotNaN();

        // Weight and bias gradients should also be set
        for (Value p : n.parameters()) {
            assertThat(p.getGrad()).isNotNaN();
        }
    }

    @Test
    void applyComputesCorrectForwardPass() {
        // Manually verify: out = tanh(w0*x0 + w1*x1 + bias)
        Neuron n = new Neuron(2);
        Value[] params = n.parameters();
        double w0 = params[0].getData();
        double w1 = params[1].getData();
        double bias = params[2].getData();

        Value[] x = { value(0.5), value(-0.3) };
        Value out = n.apply(x);

        double expectedAct = w0 * 0.5 + w1 * (-0.3) + bias;
        double expectedOut = Math.tanh(expectedAct);
        assertThat(out.getData()).isCloseTo(expectedOut, within(TOL));
    }

    @Test
    void parametersReturnsSameInstances() {
        Neuron n = new Neuron(2);
        Value[] p1 = n.parameters();
        Value[] p2 = n.parameters();
        // Each call returns a new array but containing the same Value references
        assertThat(p1).isNotSameAs(p2);
        for (int i = 0; i < p1.length; i++) {
            assertThat(p1[i]).isSameAs(p2[i]);
        }
    }
}
