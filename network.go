package lightsout

import (
	"io/ioutil"
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	t := sinLayer{}.SerializerType()
	serializer.RegisterTypedDeserializer(t, deserializeSinLayer)
}

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
			OutputCount: 100,
		},
		&neuralnet.RescaleLayer{Scale: 2 * math.Pi * 25},
		sinLayer{},
		&neuralnet.DenseLayer{
			InputCount:  100,
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

// Solve uses the network in combination with search to
// compute a solution.
func (n *Network) Solve(s *State) []Move {
	vec := n.Net.Apply(&autofunc.Variable{Vector: StateVector(s)}).Output()
	solution, state := solutionForVec(vec, *s)
	remainder := state.Solve()
	return CleanMoves(append(remainder, solution...))
}

// RawSolution uses the network to approximate the
// probability of each move in the solution.
// This can be used to find the probability that the
// network produces the correct solution.
func (n *Network) RawSolution(s *State) map[Move]float64 {
	vec := n.Net.Apply(&autofunc.Variable{Vector: StateVector(s)}).Output()
	res := map[Move]float64{}
	for i, x := range vec {
		m := Move{Row: i / 5, Col: i % 5}
		res[m] = 1 / (1 + math.Exp(-x))
	}
	return res
}

// Save saves the network to a file.
func (n *Network) Save(path string) error {
	data, err := n.Net.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, data, 0755)
}

type sinLayer struct {
	autofunc.Sin
}

func deserializeSinLayer(d []byte) (*sinLayer, error) {
	return &sinLayer{}, nil
}

func (_ sinLayer) SerializerType() string {
	return "github.com/unixpickle/lightsout.sinLayer"
}

func (_ sinLayer) Serialize() ([]byte, error) {
	return nil, nil
}
