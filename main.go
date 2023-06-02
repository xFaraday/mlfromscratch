package main

import (
	"fmt"
	"math/rand"
)

var train = [][2]float32{
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
}
var train_count int

// determine effectiveness of model
func cost(w float32, b float32) (div float32) {
	train_count := len(train)
	var result float32
	for i, _ := range train {
		input := train[i][0]
		output := input*w + b
		errordistance := output - train[i][1]
		result += errordistance * errordistance
	}
	return result / float32(train_count)
}

func main() {
	//s1 := rand.NewSource(time.Now().UnixNano())
	//w = input of model
	s1 := rand.NewSource(69)
	r1 := rand.New(s1)
	w := r1.Float32() * 10.0
	b := r1.Float32() * 5.0
	var eps float32 = 1e-3
	var rate float32 = 1e-3

	for i := 0; i < 100000; i++ {
		dcost := (cost(w+eps, b) - cost(w, b)) / eps
		dcostb := (cost(w, b+eps) - cost(w, b)) / eps
		w -= rate * dcost
		b -= rate * dcostb
		fmt.Printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b)
	}
	println("w dawg:")
	println(w)
	println("b dawg:")
	println(b)
}
