package lightsout

import (
	"log"
	"math/rand"
	"runtime"

	"github.com/unixpickle/weakai/idtrees"
)

// Tree uses decision trees to solve States.
type Tree struct {
	moveTrees []*idtrees.Tree
}

func TrainTree(verbose bool) *Tree {
	if verbose {
		log.Println("Generating all states...")
	}
	samples := allTrainingSamples()
	for i := range samples {
		samples[i] = samples[i+rand.Intn(len(samples)-i)]
	}
	validation := samples[30000:40000]
	training := samples[:30000]

	attrs := make([]idtrees.Attr, 25)
	for i := range attrs {
		attrs[i] = i
	}
	var res Tree
	for move := 0; move < 25; move++ {
		if verbose {
			log.Println("Building tree for move", move)
		}
		for _, x := range samples {
			x.MoveIdx = move
		}
		sampleSlice := make([]idtrees.Sample, len(training))
		for i, x := range training {
			sampleSlice[i] = x
		}
		tree := idtrees.ID3(sampleSlice, attrs, runtime.GOMAXPROCS(0))
		res.moveTrees = append(res.moveTrees, tree)
		for i := 0; i < 10; i++ {
			sample := validation[i]
			sample.MoveIdx = move
		}
	}

	if verbose {
		var avgWrong float64
		for _, sample := range validation {
			num := res.solutionNum(sample.State)
			diff := num ^ sample.Solution
			for i := 0; i < 25; i++ {
				if diff&(1<<uint32(i)) != 0 {
					avgWrong++
				}
			}
		}
		log.Printf("Average of %f incorrect moves", avgWrong/float64(len(validation)))
	}

	return &res
}

func (t *Tree) solutionNum(s State) uint32 {
	var res uint32
	for i, t := range t.moveTrees {
		sample := &stateAttrMap{State: s}
		if t.Classify(sample)[true] >= 0.5 {
			res |= 1 << uint32(i)
		}
	}
	return res
}

func allTrainingSamples() []*stateAttrMap {
	var res []*stateAttrMap

	visited := make([]bool, 1<<(BoardSize*BoardSize))
	queue := []*exploreNode{{State: State(0)}}
	for len(queue) > 0 {
		popped := queue[0]
		queue = queue[1:]

		res = append(res, &stateAttrMap{
			State:    popped.State,
			Solution: popped.Moves,
		})

		for row := 0; row < BoardSize; row++ {
			for col := 0; col < BoardSize; col++ {
				next := popped.State
				m := Move{row, col}
				next.Move(m)
				if !visited[int(next)] {
					visited[int(next)] = true
					moves := popped.Moves | (1 << uint32(row*BoardSize+col))
					newNode := &exploreNode{Moves: moves, State: next}
					queue = append(queue, newNode)
				}
			}
		}
	}

	return res
}

type exploreNode struct {
	State State
	Moves uint32
}

type stateAttrMap struct {
	State    State
	Solution uint32
	MoveIdx  int
}

func (s *stateAttrMap) Attr(attr idtrees.Attr) idtrees.Val {
	return s.State&(1<<uint32(attr.(int))) != 0
}

func (s *stateAttrMap) Class() idtrees.Class {
	return s.Solution&(1<<uint32(s.MoveIdx)) != 0
}
