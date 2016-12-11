package lightsout

import (
	"io/ioutil"
	"log"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
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
			OutputCount: 200,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  200,
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
	trainNet := neuralnet.Network{&trainingFunc{Solver: n.Net}}
	g := &neuralnet.BatchRGradienter{
		Learner:  trainNet.BatchLearner(),
		CostFunc: &neuralnet.AbsCost{},
	}
	precond := &sgd.Adam{Gradienter: g}
	s := generateTrainingSamples()
	batchSize := 10
	var avgCost float64
	var idx int
	sgd.SGDMini(precond, s, 0.001, batchSize, func(batch sgd.SampleSet) bool {
		cost := neuralnet.TotalCost(g.CostFunc, trainNet, batch) / float64(batchSize)
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

type trainingFunc struct {
	Solver neuralnet.Network
}

func (t *trainingFunc) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.Pool(in, func(in autofunc.Result) autofunc.Result {
		solution := t.Solver.Apply(in)
		return SoftMover{}.Apply(autofunc.Concat(solution, in))
	})
}

func (t *trainingFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(in, func(in autofunc.RResult) autofunc.RResult {
		solution := t.Solver.ApplyR(rv, in)
		return SoftMover{}.ApplyR(rv, autofunc.ConcatR(solution, in))
	})
}

func (t *trainingFunc) SerializerType() string {
	panic("not implemented")
}

func (t *trainingFunc) Serialize() ([]byte, error) {
	panic("not implemented")
}

func (t *trainingFunc) Parameters() []*autofunc.Variable {
	return t.Solver.Parameters()
}

func generateTrainingSamples() sgd.SampleSet {
	var res sgd.SliceSampleSet
	desiredOut := make(linalg.Vector, 25)
	for i := 0; i < 500000; i++ {
		start := State(0)
		for i := 0; i < 50; i++ {
			start.Move(Move{rand.Intn(BoardSize), rand.Intn(BoardSize)})
		}
		vec := StateVector(&start)
		res = append(res, neuralnet.VectorSample{Input: vec, Output: desiredOut})
	}
	return res
}
