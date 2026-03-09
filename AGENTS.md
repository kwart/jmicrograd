# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

jmicrograd is a Java port of Andrej Karpathy's micrograd — a minimal autograd engine implementing backpropagation over a dynamically-built computation graph (DAG). It supports automatic differentiation for scalar values.

## Build Commands

```bash
mvn compile          # Compile
mvn test             # Run all tests
mvn package          # Build the JAR (target/jmicrograd.jar)
mvn exec:java        # Run the App main class
```

## Architecture

- **`Value`** (`com.github.kwart.jmicrograd.Value`): Core autograd primitive. Wraps a scalar `double` and tracks computation history. Calling `backward()` performs reverse-mode autodiff via topological sort of the DAG.
- **`App`**: Demo entry point showing basic usage.

## Technical Notes

- Requires **Java 21** (`maven.compiler.release=21`).
- Uses JUnit Jupiter 6 and AssertJ for testing.
- Backpropagation functions are stored as `Function<Value, Void>` lambdas, with static constants for common ops to avoid repeated allocation.
