use std::io::Cursor;
use std::sync::Arc;

use cozy_chess::{Board, Color, File, Move, Piece, Rank, Square};

use self::layers::BitLinear;

use super::bm_runner::ab_runner;

mod layers;
mod model;

use model::{Nnue, INPUT, MID};

const NN_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/eval.bin"));

#[derive(Debug, Clone, Copy)]
pub struct Accumulator {
    accumulator: [[i16; MID]; Color::NUM]
}

fn halfka_feature(
    perspective: Color,
    king: Square,
    color: Color,
    piece: Piece,
    square: Square,
) -> usize {
    let (king, square, color) = match perspective {
        Color::White => (king, square, color),
        Color::Black => (king.flip_rank(), square.flip_rank(), !color),
    };
    let mut index = 0;
    index = index * Square::NUM + king as usize;
    index = index * Color::NUM + color as usize;
    index = index * Piece::NUM + piece as usize;
    index = index * Square::NUM + square as usize;
    index
}

impl Accumulator {
    pub fn new(ft: &BitLinear<INPUT, MID>) -> Self {
        let mut accumulator = Self { accumulator: [[0; MID]; Color::NUM] };
        accumulator.zero(ft);
        accumulator
    }

    pub fn zero(&mut self, ft: &BitLinear<INPUT, MID>) {
        self.accumulator[Color::White as usize].clone_from_slice(&ft.biases);
        self.accumulator[Color::Black as usize].clone_from_slice(&ft.biases);
    }

    pub fn update<const INCR: bool>(
        &mut self,
        ft: &BitLinear<INPUT, MID>,
        w_king: Square,
        b_king: Square,
        sq: Square,
        piece: Piece,
        color: Color,
    ) {
        let w_index = halfka_feature(Color::White, w_king, color, piece, sq);
        let b_index = halfka_feature(Color::Black, b_king, color, piece, sq);

        if INCR {
            ft.modify_feature::<1>(w_index, &mut self.accumulator[Color::White as usize]);
            ft.modify_feature::<1>(b_index, &mut self.accumulator[Color::Black as usize]);
        } else {
            ft.modify_feature::<-1>(w_index, &mut self.accumulator[Color::White as usize]);
            ft.modify_feature::<-1>(b_index, &mut self.accumulator[Color::Black as usize]);
        }
    }
}

#[derive(Debug, Clone)]
pub struct NnueState {
    nnue: Arc<Nnue>,
    accumulator: Vec<Accumulator>,
    head: usize
}

impl NnueState {
    pub fn new() -> Self {
        let nnue = Nnue::read_from(&mut Cursor::new(NN_BYTES)).unwrap();
        let nnue = Arc::new(nnue);
        let accumulator = vec![
            Accumulator::new(&nnue.ft);
            ab_runner::MAX_PLY as usize + 1
        ];

        Self {
            nnue,
            accumulator,
            head: 0,
        }
    }

    pub fn reset(&mut self, board: &Board) {
        let w_king = board.king(Color::White);
        let b_king = board.king(Color::Black);
        let acc = &mut self.accumulator[self.head];

        acc.zero(&self.nnue.ft);

        for sq in board.occupied() {
            let piece = board.piece_on(sq).unwrap();
            let color = board.color_on(sq).unwrap();
            acc.update::<true>(&self.nnue.ft, w_king, b_king, sq, piece, color);
        }
    }

    pub fn full_reset(&mut self, board: &Board) {
        self.head = 0;
        self.reset(board);
    }

    fn push_accumulator(&mut self) {
        self.accumulator.copy_within(self.head..self.head + 1, self.head + 1);
        self.head += 1;
    }

    pub fn null_move(&mut self) {
        self.push_accumulator();
    }

    pub fn make_move(&mut self, board: &Board, make_move: Move) {
        self.push_accumulator();
        let from_sq = make_move.from;
        let from_type = board.piece_on(from_sq).unwrap();
        let stm = board.side_to_move();
        let w_king = board.king(Color::White);
        let b_king = board.king(Color::Black);
        if from_type == Piece::King {
            let mut board_clone = board.clone();
            board_clone.play_unchecked(make_move);
            self.reset(&board_clone);
            return;
        }
        let acc = &mut self.accumulator[self.head];

        acc.update::<false>(&self.nnue.ft, w_king, b_king, from_sq, from_type, stm);

        let to_sq = make_move.to;
        if let Some((captured, color)) = board.piece_on(to_sq).zip(board.color_on(to_sq)) {
            acc.update::<false>(&self.nnue.ft, w_king, b_king, to_sq, captured, color);
        }

        if let Some(ep) = board.en_passant() {
            let (stm_fifth, stm_sixth) = match stm {
                Color::White => (Rank::Fifth, Rank::Sixth),
                Color::Black => (Rank::Fourth, Rank::Third),
            };
            if from_type == Piece::Pawn && to_sq == Square::new(ep, stm_sixth) {
                acc.update::<false>(
                    &self.nnue.ft,
                    w_king,
                    b_king,
                    Square::new(ep, stm_fifth),
                    Piece::Pawn,
                    !stm,
                );
            }
        }
        if Some(stm) == board.color_on(to_sq) {
            let stm_first = match stm {
                Color::White => Rank::First,
                Color::Black => Rank::Eighth,
            };
            if to_sq.file() > from_sq.file() {
                acc.update::<true>(
                    &self.nnue.ft,
                    w_king,
                    b_king,
                    Square::new(File::G, stm_first),
                    Piece::King,
                    stm,
                );
                acc.update::<true>(
                    &self.nnue.ft,
                    w_king,
                    b_king,
                    Square::new(File::F, stm_first),
                    Piece::Rook,
                    stm,
                );
            } else {
                acc.update::<true>(
                    &self.nnue.ft,
                    w_king,
                    b_king,
                    Square::new(File::C, stm_first),
                    Piece::King,
                    stm,
                );
                acc.update::<true>(
                    &self.nnue.ft,
                    w_king,
                    b_king,
                    Square::new(File::D, stm_first),
                    Piece::Rook,
                    stm,
                );
            }
        } else {
            acc.update::<true>(
                &self.nnue.ft,
                w_king,
                b_king,
                to_sq,
                make_move.promotion.unwrap_or(from_type),
                stm,
            );
        }
    }

    pub fn unmake_move(&mut self) {
        self.head -= 1;
    }

    #[inline]
    pub fn feed_forward(&mut self, stm: Color) -> i16 {
        let acc = &mut self.accumulator[self.head];
        let mut incr = [[0; MID]; Color::NUM];
        layers::sq_clipped_relu(&acc.accumulator[stm as usize], &mut incr[0]);
        layers::sq_clipped_relu(&acc.accumulator[!stm as usize], &mut incr[1]);
        layers::out(self.nnue.l1.forward(bytemuck::cast_ref(&incr))[0])
    }
}
