use tch::{
    Tensor, 
    IndexOp
};

#[allow(non_snake_case)] 
pub struct CRF {
    pub W: Tensor,
    pub n: i64,
}

impl CRF {
    /**
     * n is count of tags.
     */
    #[allow(non_snake_case)] 
    pub fn new(n: i64) -> Self {
        let n = n + 2;
        let opts = (tch::Kind::Float, tch::Device::Cpu);
        let W = Tensor::rand(&[n, n], opts);

        let mut t = W.i((.., n - 1));
        let u = t.shallow_clone();
        std::ops::SubAssign::sub_assign(&mut t, u); // W[:, n + 1] = 0
        std::ops::AddAssign::add_assign(&mut t, -10_000); // W[:, n + 1] = -1000

        let mut t = W.i((n - 2, ..));
        let u = t.shallow_clone();
        std::ops::SubAssign::sub_assign(&mut t, u); // W[n + 2, :] = 0
        std::ops::AddAssign::add_assign(&mut t, -10_000); // W[:, n + 1] = -1000
    
        Self {
            W,
            n,
        }
    }

    pub fn forward(&self, x_seq: &Tensor, init_vit_vars: &Tensor) -> (f64, Tensor) {
        self.vitervi_decode(x_seq, init_vit_vars)
    }

    fn vitervi_decode(&self, feats: &Tensor, init_vit_vars: &Tensor) -> (f64, Tensor) {
        let mut backpointers: Vec<Tensor> = Vec::default();
        let mut forward_var = init_vit_vars.shallow_clone();
        let (x, _) = feats.size2().unwrap();

        for i in 0..x {
            let mut bptrs_t: Vec<i64> = Vec::default();
            let mut viterbivars_t: Vec<f64> = Vec::default();

            for next_tag in 0..self.n {
                let next_tag_var = &forward_var + self.W.i(next_tag);
                let best_tag_id = next_tag_var.argmax(0, false);
                let best_tag_id = i64::from(best_tag_id);
             
                bptrs_t.push(best_tag_id);
                viterbivars_t.push(
                    f64::from(next_tag_var.i(best_tag_id))
                );
            }

            forward_var = Tensor::of_slice(&viterbivars_t) + &feats.i(i);
            forward_var = forward_var;

            backpointers.push(Tensor::of_slice(&bptrs_t));
        }

        let terminal_var = forward_var + self.W.i(self.n-1);
        let best_tag_id = terminal_var.argmax(0, false);
        let best_tag_id = i64::from(best_tag_id);
        let path_score = terminal_var.i(best_tag_id);
        let mut best_path = vec![best_tag_id];

        backpointers.reverse();

        for bptrs_t in backpointers {
            let best_tag_id = i64::from(bptrs_t.i(best_tag_id));
            best_path.push(best_tag_id)
        }

        let _start = best_path.pop();
        best_path.reverse();
        
        (f64::from(path_score), Tensor::of_slice(&best_path))
    }
}

mod test {
    use tch::Tensor;

    #[test]
    fn forward() {
        let mut crf = super::CRF::new(3); // B, I, O, <START>, <END>

        crf.W = Tensor::of_slice(&[
            -1.2, -1.1, 1.8, 2.2, -1.4,
            3.1, 2.0, -0.3, -0.4, -0.5,
            1.7, -0.2, 1.8, 1.1, -1.1,
            -0.5, -0.1, -0.9, -1.1, -0.4,
            1.1, 1.3, 2.1, -0.4, -0.2
        ]).view([5, 5]);
        
        let mut init_vvars: Vec<i64> = (0..crf.n)
            .map(|_| -10_000)
            .collect::<Vec<_>>();
        init_vvars[crf.n as usize - 2] = 0;
        
        let feats = Tensor::of_slice(&[
            3.7, 1.4, 1.2, -0.1, -1.2,
            2.6, 4.1, 0.9, -0.7, -2.1,
            0.2, 0.5, 2.4, -0.4, -0.8,
            3.6, 0.4, 1.1, -0.2, -0.5,
        ]).view([4, 5]);

        let (_score, path) = crf.forward(&feats, &Tensor::of_slice(&init_vvars));

        assert_eq!(path, Tensor::of_slice(&[0, 1, 2, 0]));
    }
}