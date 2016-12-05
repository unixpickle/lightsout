package lightsout

import (
	"math/rand"
	"testing"
)

func TestSolve(t *testing.T) {
	start := NewStateFlags([]bool{
		false, false, true, false, false,
		false, true, true, true, false,
		false, false, true, false, false,
		false, false, false, false, false,
		false, false, false, false, false,
	})
	solution := start.Solve()
	if solution == nil {
		t.Error("could not solve state")
	} else if len(solution) != 1 {
		t.Error("bad solution length:", len(solution))
	} else if (solution[0] != Move{1, 2}) {
		t.Errorf("bad solution: %v", solution)
	}

	start = NewStateFlags([]bool{
		false, false, true, false, false,
		false, true, true, true, false,
		false, false, true, false, false,
		false, false, false, false, true,
		false, false, false, true, true,
	})
	solution = start.Solve()
	if solution == nil {
		t.Error("could not solve state")
	} else if len(solution) != 2 {
		t.Error("bad solution length:", len(solution))
	} else {
		start.Move(solution[0])
		start.Move(solution[1])
		if !start.Solved() {
			t.Errorf("bad solution: %v", solution)
		}
	}

	for scrambleLen := 3; scrambleLen < 8; scrambleLen++ {
		start = State(0)
		for i := 0; i < scrambleLen; i++ {
			start.Move(Move{rand.Intn(BoardSize), rand.Intn(BoardSize)})
		}
		solution = start.Solve()
		if len(solution) > scrambleLen {
			t.Errorf("solution is too long: %v (max %d)", solution, scrambleLen)
		} else {
			startBackup := start
			for _, x := range solution {
				start.Move(x)
			}
			if !start.Solved() {
				t.Errorf("bad solution: %v for board %v", solution, startBackup)
			}
		}
	}
}

func BenchmarkSolve(b *testing.B) {
	// Attempt to solve an unsolvable state.
	board := State(1)
	for i := 0; i < b.N; i++ {
		board.Solve()
	}
}
