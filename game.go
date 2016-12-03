package lightsout

const BoardSize = 5

// A Move represents a move in a game of Lights Out, where
// a move consists of flipping a square and its neighbors.
type Move struct {
	Row int
	Col int
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
	s.toggle(m.Row, m.Col-1)
	s.toggle(m.Row, m.Col+1)
	s.toggle(m.Row+1, m.Col)
	s.toggle(m.Row-1, m.Col)
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
	if s.Solved() {
		return []Move{}
	}
	visited := map[State]bool{}
	queue := []solveNode{{*s, []Move{}}}
	for len(queue) > 0 {
		popped := queue[0]
		queue = queue[1:]
		for row := 0; row < BoardSize; row++ {
			for col := 0; col < BoardSize; col++ {
				next := popped.State
				m := Move{row, col}
				next.Move(m)
				if !visited[next] {
					visited[next] = true
					newMoves := append([]Move{}, popped.Prior...)
					newMoves = append(newMoves, m)
					if next.Solved() {
						return newMoves
					}
					queue = append(queue, solveNode{next, newMoves})
				}
			}
		}
	}
	return nil
}

func (s *State) toggle(row, col int) {
	if row < 0 || row >= BoardSize {
		return
	}
	if col < 0 || col >= BoardSize {
		return
	}
	*s ^= (1 << (uint(row)*BoardSize + uint(col)))
}

type solveNode struct {
	State State
	Prior []Move
}
