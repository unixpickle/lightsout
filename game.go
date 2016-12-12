package lightsout

import "strings"

const BoardSize = 5

// A Move represents a move in a game of Lights Out, where
// a move consists of flipping a square and its neighbors.
type Move struct {
	Row int
	Col int
}

// CleanMoves simplifies a set of moves by removing
// duplicates and sorting the moves in a canonical order.
func CleanMoves(m []Move) []Move {
	moveFlags := make([]bool, BoardSize*BoardSize)
	for _, move := range m {
		idx := move.Row*BoardSize + move.Col
		moveFlags[idx] = !moveFlags[idx]
	}
	res := make([]Move, 0, len(moveFlags))
	for i, x := range moveFlags {
		if x {
			res = append(res, Move{Row: i / BoardSize, Col: i % BoardSize})
		}
	}
	return res
}

// A State stores the instantaneous state for a game.
// It is a bitmap, where the bit corresponding to (1<<0)
// is the first bit in the map, (1<<1) the second, etc.
// The state is stored in row-major order, with a 0 value
// indicating an unlit square.
type State uint32

// NewStateFlags creates a state from a slice of flags
// representing a bitmap.
func NewStateFlags(f []bool) State {
	res := State(0)
	for i, x := range f {
		if x {
			res |= (1 << uint(i))
		}
	}
	return res
}

// Move applies a move to the given square, flipping all
// of its neighbors.
func (s *State) Move(m Move) {
	s.toggle(m.Row, m.Col)
	if m.Col > 0 {
		s.toggle(m.Row, m.Col-1)
	}
	if m.Col+1 < BoardSize {
		s.toggle(m.Row, m.Col+1)
	}
	if m.Row > 0 {
		s.toggle(m.Row-1, m.Col)
	}
	if m.Row+1 < BoardSize {
		s.toggle(m.Row+1, m.Col)
	}
}

// Get gets the square value at the given row and column.
func (s *State) Get(row, col int) bool {
	return 0 != (*s & (1 << (uint(row)*BoardSize + uint(col))))
}

// Solved returns true if all the lights are out.
func (s *State) Solved() bool {
	return 0 == (*s & 0x1FFFFFF)
}

// Solve finds an optimal solution to the state or returns
// nil if the state is unsolvable.
func (s *State) Solve() []Move {
	return recursiveSolve(*s, 0, computeMaxMoves(), BoardSize*BoardSize)
}

func (s *State) String() string {
	var res []string
	for i := 0; i < 5; i++ {
		var row []string
		for j := 0; j < 5; j++ {
			if s.Get(i, j) {
				row = append(row, "1")
			} else {
				row = append(row, "0")
			}
		}
		res = append(res, strings.Join(row, " "))
	}
	return "[" + strings.Join(res, "; ") + "]"
}

func (s *State) toggle(row, col int) {
	*s ^= (1 << (uint(row)*BoardSize + uint(col)))
}

func recursiveSolve(s State, moveIdx int, maxMoves []int, maxLen int) []Move {
	if s.Solved() {
		return []Move{}
	} else if moveIdx == BoardSize*BoardSize || maxLen == 0 {
		return nil
	}
	for spot, max := range maxMoves {
		if moveIdx > max {
			if s&(1<<uint32(spot)) != 0 {
				return nil
			}
		}
	}
	solution1 := recursiveSolve(s, moveIdx+1, maxMoves, maxLen)
	m := Move{Row: moveIdx / BoardSize, Col: moveIdx % BoardSize}
	s.Move(m)
	solution2 := recursiveSolve(s, moveIdx+1, maxMoves, maxLen-1)
	if solution2 != nil {
		solution2 = append(solution2, m)
	} else {
		return solution1
	}
	if solution1 == nil || len(solution2) < len(solution1) {
		return solution2
	}
	return solution1
}

// computeMaxMoves computes the maximum move index which
// affects each piece on a board.
func computeMaxMoves() []int {
	res := make([]int, BoardSize*BoardSize)
	for i := range res {
		for j := 0; j < BoardSize*BoardSize; j++ {
			m := Move{Row: j / BoardSize, Col: j % BoardSize}
			s := State(0)
			s.Move(m)
			if s&(1<<uint32(i)) != 0 {
				res[i] = j
			}
		}
	}
	return res
}
