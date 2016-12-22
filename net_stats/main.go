package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/lightsout"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 2 {
		fmt.Println("Usage: net_solve <network>")
		os.Exit(1)
	}
	net, err := lightsout.LoadNetwork(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load network:", err)
		os.Exit(1)
	}
	scramble := lightsout.State(0)
	for i := 0; i < 100; i++ {
		scramble.Move(lightsout.Move{Row: rand.Intn(5), Col: rand.Intn(5)})
	}
	solution := net.Solve(&scramble)
	probs := net.RawSolution(&scramble)
	totalProb := 1.0
	for m, prob := range probs {
		inSolution := false
		for _, y := range solution {
			if y == m {
				inSolution = true
			}
		}
		if inSolution {
			totalProb *= prob
		} else {
			totalProb *= 1 - prob
		}
	}
	fmt.Println("Solution probability:", totalProb)
}
