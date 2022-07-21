use cozy_chess::{Board, Color, Move, Piece, Square};

pub const MAX_VALUE: i32 = 512;

/// From-to table
type ButterflyTable<T> = [[T; Square::NUM]; Square::NUM];
fn butterfly_table<T: Copy>(value: T) -> ButterflyTable<T> {
    [[value; Square::NUM]; Square::NUM]
}

type PieceToSquareTable<T> = [[T; Square::NUM]; Piece::NUM];
fn piece_to_square_table<T: Copy>(value: T) -> PieceToSquareTable<T> {
    [[value; Square::NUM]; Piece::NUM]
}

#[derive(Debug, Clone)]
pub struct HistoryTable {
    table: Box<[ButterflyTable<i16>; Color::NUM]>,
}

impl HistoryTable {
    pub fn new() -> Self {
        Self {
            table: Box::new([butterfly_table(0); Color::NUM]),
        }
    }

    pub fn get(&self, color: Color, from: Square, to: Square) -> i16 {
        self.table[color as usize][from as usize][to as usize]
    }

    fn get_mut(&mut self, color: Color, from: Square, to: Square) -> &mut i16 {
        &mut self.table[color as usize][from as usize][to as usize]
    }

    pub fn cutoff(&mut self, board: &Board, make_move: Move, fails: &[Move], amt: u32) {
        let color = board.side_to_move();

        let value = self.get(color, make_move.from, make_move.to);
        let change = (amt * amt) as i16;
        let decay = (change as i32 * value as i32 / MAX_VALUE) as i16;

        let increment = change - decay;

        *self.get_mut(color, make_move.from, make_move.to) += increment;

        for &quiet in fails {
            let value = self.get(color, quiet.from, quiet.to);
            let decay = (change as i32 * value as i32 / MAX_VALUE) as i16;
            let decrement = change + decay;

            *self.get_mut(color, quiet.from, quiet.to) -= decrement;
        }
    }
}

#[derive(Debug, Clone)]
pub struct CounterMoveTable {
    table: Box<[PieceToSquareTable<Option<Move>>; Color::NUM]>,
}

impl CounterMoveTable {
    pub fn new() -> Self {
        Self {
            table: Box::new([piece_to_square_table(None); Color::NUM]),
        }
    }

    pub fn get(&self, color: Color, piece: Piece, to: Square) -> Option<Move> {
        self.table[color as usize][piece as usize][to as usize]
    }

    pub fn cutoff(&mut self, board: &Board, prev_move: Move, cutoff_move: Move, amt: u32) {
        if amt > 20 {
            return;
        }
        let to = prev_move.to;
        let color = board.side_to_move();
        let piece = board.piece_on(to).unwrap_or(Piece::King);
        self.table[color as usize][piece as usize][to as usize] = Some(cutoff_move);
    }
}

#[derive(Debug, Clone)]
pub struct DoubleMoveHistory {
    table: Box<[PieceToSquareTable<PieceToSquareTable<i16>>; Color::NUM]>,
}

impl DoubleMoveHistory {
    pub fn new() -> Self {
        Self {
            table: Box::new([piece_to_square_table(piece_to_square_table(0)); Color::NUM]),
        }
    }

    pub fn get(
        &self,
        color: Color,
        piece_0: Piece,
        to_0: Square,
        piece_1: Piece,
        to_1: Square,
    ) -> i16 {
        self.table[color as usize][piece_0 as usize][to_0 as usize][piece_1 as usize][to_1 as usize]
    }

    fn get_mut(
        &mut self,
        color: Color,
        piece_0: Piece,
        to_0: Square,
        piece_1: Piece,
        to_1: Square,
    ) -> &mut i16 {
        &mut self.table[color as usize][piece_0 as usize][to_0 as usize][piece_1 as usize]
            [to_1 as usize]
    }

    pub fn cutoff(
        &mut self,
        board: &Board,
        prev_move: Move,
        make_move: Move,
        fails: &[Move],
        amt: u32,
    ) {
        let color = board.side_to_move();

        let prev_piece = board.piece_on(prev_move.to).unwrap_or(Piece::King);
        let piece = board.piece_on(make_move.from).unwrap();

        let value = self.get(color, prev_piece, prev_move.to, piece, make_move.to);
        let change = (amt * amt) as i16;
        let decay = (change as i32 * value as i32 / MAX_VALUE) as i16;

        let increment = change - decay;

        *self.get_mut(color, prev_piece, prev_move.to, piece, make_move.to) += increment;

        for &quiet in fails {
            let piece = board.piece_on(quiet.from).unwrap();
            let value = self.get(color, prev_piece, prev_move.to, piece, quiet.to);
            let decay = (change as i32 * value as i32 / MAX_VALUE) as i16;
            let decrement = change + decay;

            *self.get_mut(color, prev_piece, prev_move.to, piece, quiet.to) -= decrement;
        }
    }
}
