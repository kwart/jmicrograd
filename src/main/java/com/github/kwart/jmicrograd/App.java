package com.github.kwart.jmicrograd;

import static com.github.kwart.jmicrograd.Value.value;

import java.util.Arrays;

/**
 * The App!
 */
public class App {

    public static void main(String[] args) {
        // Original demo
        Value a = new Value(-2.);
        Value b = new Value(3.);
        Value d = a.mul(b);
        Value e = a.add(b);
        Value f = d.mul(e);
        f.backward();
        System.out.println("a: " + a);
        System.out.println("b: " + b);

        System.out.println();

        // MLP demo (ported from micrograd)
        Value[] x = { value(2.0), value(3.0), value(-1.0) };
        MLP n = new MLP(3, new int[]{4, 4, 1});
        System.out.println("Single forward pass: " + Arrays.toString(n.apply(x)));

        Value[][] xs = {
            { value(2.0), value(3.0), value(-1.0) },
            { value(3.0), value(-1.0), value(0.5) },
            { value(0.5), value(1.0), value(1.0) },
            { value(1.0), value(1.0), value(-1.0) },
        };
        double[] ys = { 1.0, -1.0, -1.0, 1.0 }; // desired targets

        for (int k = 0; k < 20; k++) {
            // forward pass
            Value[] ypred = new Value[xs.length];
            for (int i = 0; i < xs.length; i++) {
                ypred[i] = n.apply(xs[i])[0];
            }

            // loss = sum((yout - ygt)**2)
            Value loss = value(0.0);
            for (int i = 0; i < ys.length; i++) {
                loss = loss.add(ypred[i].add(value(-ys[i])).pow(2));
            }

            // backward pass
            n.zeroGrad();
            loss.backward();

            // update
            for (Value p : n.parameters()) {
                p.setData(p.getData() + (-0.1) * p.getGrad());
            }

            System.out.println(k + " " + loss.getData());
        }
    }
}
