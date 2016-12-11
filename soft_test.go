package lightsout

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestSoftMover(t *testing.T) {
	input := &autofunc.Variable{Vector: make(linalg.Vector, 25)}
	for i := 0; i < 25; i++ {
		if rand.Intn(2) != 0 {
			input.Vector[i] = 1
		}
	}

	moves := &autofunc.Variable{Vector: make(linalg.Vector, 25)}
	for i := range input.Vector {
		moves.Vector[i] = rand.NormFloat64()
	}
	sm := autofunc.Softmax{}
	probs := sm.Apply(moves).Output()

	input.Vector = append(probs, input.Vector...)
	rvec := autofunc.RVector{input: make(linalg.Vector, len(input.Vector))}
	for i := range input.Vector {
		rvec[input][i] = rand.NormFloat64()
	}

	checker := functest.RFuncChecker{
		F:     SoftMover{},
		Input: input,
		Vars:  []*autofunc.Variable{input},
		RV:    rvec,
	}
	checker.FullCheck(t)
}

func TestSoftSolve(t *testing.T) {
	if testing.Short() {
		t.Skip("test will take too long")
	}
	start := State(0)
	for i := 0; i < 10; i++ {
		start.Move(Move{rand.Intn(BoardSize), rand.Intn(BoardSize)})
	}
	for i := 0; i < 100; i++ {
		if len(SoftSolve(&start)) > 0 {
			return
		}
	}
	t.Error("no solution found")
}
