package main

import (
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/lightsout"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	var net *lightsout.Network
	var err error
	if net, err = lightsout.LoadNetwork("out_net"); err != nil {
		net = lightsout.NewNetwork()
	}

	s := allTrainingSamples()
	g := &sgd.Adam{
		Gradienter: &neuralnet.BatchRGradienter{
			Learner:  net.Net.BatchLearner(),
			CostFunc: neuralnet.SigmoidCECost{},
		},
	}
	var idx int
	bs := 5000
	var lastB sgd.SampleSet
	sgd.SGDMini(g, s, 0.0001, bs, func(batch sgd.SampleSet) bool {
		var last float64
		if lastB != nil {
			last = neuralnet.TotalCost(neuralnet.SigmoidCECost{}, net.Net,
				lastB) / float64(bs)
		}
		lastB = batch.Copy()
		log.Printf("batch %d: cost=%f last=%f", idx,
			neuralnet.TotalCost(neuralnet.SigmoidCECost{}, net.Net, batch)/float64(bs),
			last)
		idx++
		return true
	})

	net.Save("out_net")
}

type trainingSamples struct {
	scrambles []lightsout.State
	solutions []uint32
}

func allTrainingSamples() *trainingSamples {
	res := &trainingSamples{}

	visited := make([]bool, 1<<(lightsout.BoardSize*lightsout.BoardSize))
	queue := []*exploreNode{{State: lightsout.State(0)}}
	for len(queue) > 0 {
		popped := queue[0]
		queue = queue[1:]

		res.scrambles = append(res.scrambles, popped.State)
		res.solutions = append(res.solutions, popped.Moves)

		for row := 0; row < lightsout.BoardSize; row++ {
			for col := 0; col < lightsout.BoardSize; col++ {
				next := popped.State
				m := lightsout.Move{row, col}
				next.Move(m)
				if !visited[int(next)] {
					visited[int(next)] = true
					moves := popped.Moves | (1 << uint32(row*lightsout.BoardSize+col))
					newNode := &exploreNode{Moves: moves, State: next}
					queue = append(queue, newNode)
				}
			}
		}
	}

	return res
}

func (t *trainingSamples) Len() int {
	return len(t.scrambles)
}

func (t *trainingSamples) Swap(i, j int) {
	t.scrambles[i], t.scrambles[j] = t.scrambles[j], t.scrambles[i]
	t.solutions[i], t.solutions[j] = t.solutions[j], t.solutions[i]
}

func (t *trainingSamples) GetSample(i int) interface{} {
	scramble := t.scrambles[i]
	solution := lightsout.State(t.solutions[i])
	return neuralnet.VectorSample{
		Input:  lightsout.StateVector(&scramble),
		Output: lightsout.StateVector(&solution),
	}
}

func (t *trainingSamples) Copy() sgd.SampleSet {
	return &trainingSamples{
		scrambles: append([]lightsout.State{}, t.scrambles...),
		solutions: append([]uint32{}, t.solutions...),
	}
}

func (t *trainingSamples) Subset(i, j int) sgd.SampleSet {
	return &trainingSamples{
		scrambles: t.scrambles[i:j],
		solutions: t.solutions[i:j],
	}
}

type exploreNode struct {
	State lightsout.State
	Moves uint32
}
