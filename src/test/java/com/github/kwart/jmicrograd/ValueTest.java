package com.github.kwart.jmicrograd;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static com.github.kwart.jmicrograd.Value.value;

import org.junit.jupiter.api.Test;

/**
 * Tests ported from https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
 */
class ValueTest {

    private static final double TOL = 1e-6;

    /**
     * Translated from test_sanity_check in micrograd.
     *
     * <pre>
     * x = Value(-4.0)
     * z = 2 * x + 2 + x
     * q = z.relu() + z * x
     * h = (z * z).relu()
     * y = h + q + q * x
     * y.backward()
     * </pre>
     *
     * PyTorch reference values:
     *   y.data = -20.0
     *   x.grad = 46.0
     */
    @Test
    void testSanityCheck() {
        Value x = value(-4.0);
        // z = 2 * x + 2 + x
        Value z = value(2.0).mul(x).add(value(2.0)).add(x);
        // q = z.relu() + z * x
        Value q = z.relu().add(z.mul(x));
        // h = (z * z).relu()
        Value h = z.mul(z).relu();
        // y = h + q + q * x
        Value y = h.add(q).add(q.mul(x));
        y.backward();

        // forward pass
        assertThat(y.getData()).isEqualTo(-20.0);
        // backward pass
        assertThat(x.getGrad()).isEqualTo(46.0);
    }

    /**
     * Translated from test_more_ops in micrograd.
     *
     * <pre>
     * a = Value(-4.0)
     * b = Value(2.0)
     * c = a + b
     * d = a * b + b**3
     * c += c + 1
     * c += 1 + c + (-a)
     * d += d * 2 + (b + a).relu()
     * d += 3 * d + (b - a).relu()
     * e = c - d
     * f = e**2
     * g = f / 2.0
     * g += 10.0 / f
     * g.backward()
     * </pre>
     *
     * PyTorch reference values:
     *   g.data = 24.70408163265306
     *   a.grad = 138.83381924198252
     *   b.grad = 645.5772594752186
     */
    @Test
    void testMoreOps() {
        Value a = value(-4.0);
        Value b = value(2.0);
        // c = a + b
        Value c = a.add(b);
        // d = a * b + b**3
        Value d = a.mul(b).add(b.pow(3));
        // c += c + 1  →  c = c + c + 1
        c = c.add(c).add(value(1));
        // c += 1 + c + (-a)  →  c = c + 1 + c + (-a)
        c = c.add(value(1)).add(c).add(a.mul(value(-1)));
        // d += d * 2 + (b + a).relu()  →  d = d + d*2 + relu(b+a)
        d = d.add(d.mul(value(2))).add(b.add(a).relu());
        // d += 3 * d + (b - a).relu()  →  d = d + 3*d + relu(b + (-a))
        d = d.add(value(3).mul(d)).add(b.add(a.mul(value(-1))).relu());
        // e = c - d  →  e = c + (-1)*d
        Value e = c.add(d.mul(value(-1)));
        // f = e**2
        Value f = e.pow(2);
        // g = f / 2.0  →  g = f * 0.5
        Value g = f.mul(value(0.5));
        // g += 10.0 / f  →  g = g + 10.0 * f^(-1)
        g = g.add(value(10.0).mul(f.pow(-1)));
        g.backward();

        // forward pass
        assertThat(g.getData()).isCloseTo(24.70408163265306, within(TOL));
        // backward pass
        assertThat(a.getGrad()).isCloseTo(138.83381924198252, within(TOL));
        assertThat(b.getGrad()).isCloseTo(645.5772594752186, within(TOL));
    }

    @Test
    void testTanh() {
        Value x = value(0.5);
        Value y = x.tanh();
        y.backward();

        double expected = Math.tanh(0.5);
        assertThat(y.getData()).isCloseTo(expected, within(TOL));
        // tanh'(x) = 1 - tanh(x)^2
        assertThat(x.getGrad()).isCloseTo(1 - expected * expected, within(TOL));
    }

    @Test
    void testSigmoid() {
        Value x = value(1.0);
        Value y = x.sigmoid();
        y.backward();

        double expected = 1.0 / (1.0 + Math.exp(-1.0));
        assertThat(y.getData()).isCloseTo(expected, within(TOL));
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        assertThat(x.getGrad()).isCloseTo(expected * (1 - expected), within(TOL));
    }

    @Test
    void testReluPositive() {
        Value x = value(3.0);
        Value y = x.relu();
        y.backward();

        assertThat(y.getData()).isEqualTo(3.0);
        assertThat(x.getGrad()).isEqualTo(1.0);
    }

    @Test
    void testReluNegative() {
        Value x = value(-3.0);
        Value y = x.relu();
        y.backward();

        assertThat(y.getData()).isEqualTo(0.0);
        assertThat(x.getGrad()).isEqualTo(0.0);
    }

    @Test
    void testPow() {
        Value x = value(3.0);
        Value y = x.pow(2);
        y.backward();

        assertThat(y.getData()).isEqualTo(9.0);
        // d/dx(x^2) = 2x = 6
        assertThat(x.getGrad()).isEqualTo(6.0);
    }
}
