package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

var train = [][3]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

type Xor struct {
	//neuron 1
	or_w1 float64
	or_w2 float64
	or_b  float64
	//neuron 2
	nand_w1 float64
	nand_w2 float64
	nand_b  float64
	//neuron 3
	and_w1 float64
	and_w2 float64
	and_b  float64
}

func sigmoid(x float64) (sig float64) {
	return 1.0 / (1.0 + math.Exp(-x))
}

func forward(m Xor, x1 float64, x2 float64) (res float64) {
	//1st layer
	n1 := sigmoid(m.or_w1*x1 + m.or_w2*x2 + m.or_b)
	n2 := sigmoid(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b)
	//2nd layer
	return sigmoid(n1*m.and_w1 + n2*m.and_w2 + m.and_b)
}

func learnbb(m Xor, g Xor, rate float64) (f Xor) {
	m.or_w1 -= rate * g.or_w1
	m.or_w2 -= rate * g.or_w2
	m.or_b -= rate * g.or_b
	m.nand_w1 -= rate * g.nand_w1
	m.nand_w2 -= rate * g.nand_w2
	m.nand_b -= rate * g.nand_b
	m.and_w1 -= rate * g.and_w1
	m.and_w2 -= rate * g.and_w2
	m.and_b -= rate * g.and_b
	return m
}

func finite_diff(m Xor, eps float64) (g Xor) {

	var c float64 = cost(m)
	var saved float64

	saved = m.or_w1
	m.or_w1 += eps
	g.or_w1 = (cost(m) - c) / eps
	m.or_w1 = saved

	saved = m.or_w2
	m.or_w2 += eps
	g.or_w2 = (cost(m) - c) / eps
	m.or_w2 = saved

	saved = m.or_b
	m.or_b += eps
	g.or_b = (cost(m) - c) / eps
	m.or_b = saved

	saved = m.nand_w1
	m.nand_w1 += eps
	g.nand_w1 = (cost(m) - c) / eps
	m.nand_w1 = saved

	saved = m.nand_w2
	m.nand_w2 += eps
	g.nand_w2 = (cost(m) - c) / eps
	m.nand_w2 = saved

	saved = m.nand_b
	m.nand_b += eps
	g.nand_b = (cost(m) - c) / eps
	m.nand_b = saved

	saved = m.and_w1
	m.and_w1 += eps
	g.and_w1 = (cost(m) - c) / eps
	m.and_w1 = saved

	saved = m.and_w2
	m.and_w2 += eps
	g.and_w2 = (cost(m) - c) / eps
	m.and_w2 = saved

	saved = m.and_b
	m.and_b += eps
	g.and_b = (cost(m) - c) / eps
	m.and_b = saved

	return g
}

func cost(m Xor) (div float64) {
	train_count := len(train)
	var result float64
	for i, _ := range train {
		input1 := train[i][0]
		input2 := train[i][1]
		output := forward(m, input1, input2)
		errordistance := output - train[i][2]
		result += errordistance * errordistance
	}
	return result / float64(train_count)
}

func rand_xor() (m Xor) {
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	m = Xor{
		or_w1:   r1.Float64(),
		or_w2:   r1.Float64(),
		or_b:    r1.Float64(),
		nand_w1: r1.Float64(),
		nand_w2: r1.Float64(),
		nand_b:  r1.Float64(),
		and_w1:  r1.Float64(),
		and_w2:  r1.Float64(),
		and_b:   r1.Float64(),
	}
	return m
}

func print_xor(m Xor) {
	fmt.Printf("%#v", m)
}

func main() {
	var eps float64 = 1e-1
	var rate float64 = 1e-1
	m := rand_xor()
	print_xor(m)
	fmt.Printf("\n-------------------------------------------\n")
	for i := 0; i < 100000; i++ {
		g := finite_diff(m, eps)
		m = learnbb(m, g, rate)
		fmt.Printf("cost = %f\n", cost(m))
	}

	for i := 0.0; i < 2; i++ {
		for j := 0.0; j < 2; j++ {
			fmt.Printf("%f | %f = %f\n", i, j, forward(m, i, j))
		}
	}
}
