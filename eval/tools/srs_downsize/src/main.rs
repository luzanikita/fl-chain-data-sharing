// Downsize an official halo2 KZG SRS (RawBytes) to a smaller k, the same way halo2 does it.
// usage: srs_downsize <in.srs> <k> <out.srs>
use halo2_proofs::halo2curves::bn256::Bn256;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::kzg::commitment::ParamsKZG;
use halo2_proofs::SerdeFormat;
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let k: u32 = a[2].parse().unwrap();
    let mut r = BufReader::new(File::open(&a[1]).unwrap());
    let mut p = ParamsKZG::<Bn256>::read_custom(&mut r, SerdeFormat::RawBytes).unwrap();
    p.downsize(k);
    let mut w = BufWriter::new(File::create(&a[3]).unwrap());
    p.write_custom(&mut w, SerdeFormat::RawBytes).unwrap();
    eprintln!("wrote k={} to {}", p.k(), a[3]);
}
