package com.github.kwart.jmicrograd;

/**
 * The App!
 */
public class App {

    public static void main(String[] args) {
        Value a = new Value(-2.);
        Value b = new Value(3.);
        Value d = a.mul(b);
        Value e = a.add(b);
        Value f = d.mul(e);
        f.backward();
        System.out.println("a: " + a);
        System.out.println("b: " + b);
    }
}
