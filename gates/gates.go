package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// or gate
var train = [][3]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 1},
}

func sigmoid(x float64) (sig float64) {
	return 1.0 / (1.0 + math.Exp(-x))
}

// training
func cost(w1 float64, w2 float64, b float64) (div float64) {
	train_count := len(train)
	var result float64
	for i, _ := range train {
		input1 := train[i][0]
		input2 := train[i][1]
		output := sigmoid(input1*w1 + input2*w2 + b)
		errordistance := output - train[i][2]
		result += errordistance * errordistance
	}
	return result / float64(train_count)
}

func main() {
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	w1 := r1.Float64()
	w2 := r1.Float64()
	b := r1.Float64()

	var eps float64 = 1e-2
	var rate float64 = 1e-1

	for i := 0; i < 1000000; i++ {
		bruh := cost(w1, w2, b)
		fmt.Printf("w1 = %f, w2 = %f, b = %f, c = %f\n", w1, w2, b, bruh)
		dw1 := (cost(w1+eps, w2, b) - bruh) / eps
		dw2 := (cost(w1, w2+eps, b) - bruh) / eps
		db := (cost(w1, w2, b+eps) - bruh) / eps
		w1 -= rate * dw1
		w2 -= rate * dw2
		b -= rate * db
	}

	for i := 0.0; i < 2; i++ {
		for j := 0.0; j < 2; j++ {
			fmt.Printf("%f | %f = %f\n", i, j, sigmoid(i*w1+j*w2+b))
		}
	}
}
