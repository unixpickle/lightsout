package lightsout

import (
	"io/ioutil"
	"log"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// A Network uses an artificial neural network to find
// solutions to games of lightsout.
type Network struct {
	Net neuralnet.Network
}

// NewNetwork creates a new, untrained network.
func NewNetwork() *Network {
	res := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  25,
			OutputCount: 300,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  300,
			OutputCount: 500,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  500,
			OutputCount: 25,
		},
		&neuralnet.Sigmoid{},
	}
	res.Randomize()
	return &Network{Net: res}
}

// LoadNetwork loads a network from a file.
func LoadNetwork(path string) (*Network, error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	net, err := neuralnet.DeserializeNetwork(contents)
	if err != nil {
		return nil, err
	}
	return &Network{Net: net}, nil
}

// Train trains the network.
// If the verbose flag is `true`, the progress is logged.
func (n *Network) Train(verbose bool) {
	g := &neuralnet.BatchRGradienter{
		Learner:  n.Net.BatchLearner(),
		CostFunc: &neuralnet.AbsCost{},
	}
	precond := &sgd.Adam{Gradienter: g}

	if verbose {
		log.Println("Generating training samples...")
	}
	s := allTrainingSamples()
	if verbose {
		log.Println("Training on", s.Len(), "samples...")
	}
	sgd.ShuffleSampleSet(s)

	batchSize := 10
	var avgCost float64
	var idx int
	sgd.SGDMini(precond, s, 0.001, batchSize, func(batch sgd.SampleSet) bool {
		cost := neuralnet.TotalCost(g.CostFunc, n.Net, batch) / float64(batchSize)
		if avgCost == 0 {
			avgCost = cost
		} else {
			avgCost += 0.01 * (cost - avgCost)
		}
		if verbose && idx%50 == 0 {
			log.Println("mean cost:", avgCost)
		}
		idx++
		return avgCost > 1e-1
	})
}

// Solve uses the network to solve a state.
// If the network fails to produce a correct solution,
// nil is returned.
func (n *Network) Solve(s *State) []Move {
	vec := n.Net.Apply(&autofunc.Variable{Vector: StateVector(s)}).Output()
	return solutionForVec(vec, *s)
}

// Save saves the network to a file.
func (n *Network) Save(path string) error {
	data, err := n.Net.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, data, 0755)
}

type trainingSamples struct {
	scrambles []State
	solutions []uint32
}

func allTrainingSamples() *trainingSamples {
	res := &trainingSamples{}

	visited := make([]bool, 1<<(BoardSize*BoardSize))
	queue := []*exploreNode{{State: State(0)}}
	for len(queue) > 0 {
		popped := queue[0]
		queue = queue[1:]

		res.scrambles = append(res.scrambles, popped.State)
		res.solutions = append(res.solutions, popped.Moves)

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

func (t *trainingSamples) Len() int {
	return len(t.scrambles)
}

func (t *trainingSamples) Swap(i, j int) {
	t.scrambles[i], t.scrambles[j] = t.scrambles[j], t.scrambles[i]
	t.solutions[i], t.solutions[j] = t.solutions[j], t.solutions[i]
}

func (t *trainingSamples) GetSample(i int) interface{} {
	scramble := t.scrambles[i]
	solution := State(t.solutions[i])
	return neuralnet.VectorSample{
		Input:  StateVector(&scramble),
		Output: StateVector(&solution),
	}
}

func (t *trainingSamples) Copy() sgd.SampleSet {
	return &trainingSamples{
		scrambles: append([]State{}, t.scrambles...),
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
	State State
	Moves uint32
}
