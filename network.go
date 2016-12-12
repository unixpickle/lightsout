package lightsout

import (
	"io/ioutil"

	"github.com/unixpickle/autofunc"
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
			OutputCount: 200,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  200,
			OutputCount: 200,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  200,
			OutputCount: 200,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  200,
			OutputCount: 25,
		},
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

// Solve uses the network in combination to regular search
// to compute a solution.
func (n *Network) Solve(s *State) []Move {
	vec := n.Net.Apply(&autofunc.Variable{Vector: StateVector(s)}).Output()
	solution, state := solutionForVec(vec, *s)
	remainder := state.Solve()
	return CleanMoves(append(remainder, solution...))
}

// Save saves the network to a file.
func (n *Network) Save(path string) error {
	data, err := n.Net.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, data, 0755)
}
