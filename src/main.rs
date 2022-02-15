use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};
use rand::prelude::*;
use std::time::Instant;
use structopt::StructOpt;
use ultraviolet::{f32x8, Vec3, Vec3x8};

#[derive(Debug, StructOpt, Default)]
#[structopt(name = "Fields VR", about = "Visualizes fields")]
struct Opt {
    #[structopt(flatten)]
    arrows: ArrowCfg,

    #[structopt(flatten)]
    particles: ParticleCfg,

    /// Visualize with VR
    #[structopt(long)]
    vr: bool,
}

#[derive(Debug, StructOpt, Default, Clone, Copy)]
struct ArrowCfg {
    /// Arrow density per unit volume
    #[structopt(short = "d", long, default_value = "1.0")]
    arrow_density: f32,

    // TODO: Use a sphere instead?
    /// Half of the side length of the cube in which arrows are placed.
    #[structopt(short = "r", long, default_value = "3.0")]
    arrow_radius: f32,
    // /// Base length of arrows
    // #[structopt(short = "l", long, default_value = "0.01")]
    // arrow_length: f32,
}

#[derive(Debug, StructOpt, Default, Clone, Copy)]
struct ParticleCfg {
    /// Number of parcels to simulate. Each parcel contains 8 particles.
    #[structopt(short = "n", long, default_value = "10")]
    n_parcels: usize,

    #[structopt(short = "t", long, default_value = "3.0")]
    domain_radius: f32,

    /// Particle mass
    #[structopt(short = "m", long, default_value = "1.0")]
    mass: f32,
}

fn main() -> Result<()> {
    let args = Opt::from_args();
    launch::<Opt, FieldVisualizer>(Settings::default().vr(args.vr).args(args))
}

fn field(pos: Vec3x8) -> Vec3x8 {
    let obj_pos = Vec3x8::splat(Vec3::zero());
    let diff = pos - obj_pos;
    let dist_sq = diff.mag_sq();
    let norm = diff.normalized() / dist_sq;
    norm
}

struct FieldVisualizer {
    line_vertices: VertexBuffer,
    line_indices: IndexBuffer,
    line_shader: Shader,

    point_vertices: VertexBuffer,
    point_indices: IndexBuffer,
    point_shader: Shader,

    present: FieldPresentation,

    delta_time: Instant,

    camera: MultiPlatformCamera,
}

impl App<Opt> for FieldVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, args: Opt) -> Result<Self> {
        let present = FieldPresentation::new(&args, field);

        Ok(Self {
            line_vertices: ctx.vertices(&present.line_gb.vertices, true)?,
            line_indices: ctx.indices(&present.line_gb.indices, true)?,

            line_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Lines,
            )?,

            point_vertices: ctx.vertices(&present.point_gb.vertices, true)?,
            point_indices: ctx.indices(&present.point_gb.indices, true)?,

            point_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Points,
            )?,

            present,

            delta_time: Instant::now(),

            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        let dt = self.delta_time.elapsed();
        self.delta_time = Instant::now();

        self.present.step(field, dt.as_secs_f32());

        ctx.update_indices(self.point_indices, &self.present.point_gb.indices)?;
        ctx.update_vertices(self.point_vertices, &self.present.point_gb.vertices)?;

        ctx.update_indices(self.line_indices, &self.present.line_gb.indices)?;
        ctx.update_vertices(self.line_vertices, &self.present.line_gb.vertices)?;

        Ok(vec![
            DrawCmd::new(self.point_vertices)
                .indices(self.point_indices)
                .shader(self.point_shader),
            DrawCmd::new(self.line_vertices)
                .indices(self.line_indices)
                .shader(self.line_shader),
        ])
    }

    fn event(
        &mut self,
        ctx: &mut Context,
        platform: &mut Platform,
        mut event: Event,
    ) -> Result<()> {
        if self.camera.handle_event(&mut event) {
            ctx.set_camera_prefix(self.camera.get_prefix())
        }
        idek::close_when_asked(platform, &event);
        Ok(())
    }
}

struct FieldPresentation {
    line_gb: GraphicsBuilder,
    point_gb: GraphicsBuilder,
    sim: ParticleSim,
    arrow_cfg: ArrowCfg,
}

impl FieldPresentation {
    pub fn new(args: &Opt, field: impl Fn(Vec3x8) -> Vec3x8) -> Self {
        let mut line_gb = GraphicsBuilder::new();
        let mut point_gb = GraphicsBuilder::new();

        let sim = ParticleSim::new(args.particles);

        particle_mesh(&mut point_gb, &sim);
        arrow_mesh(&mut line_gb, field, &args.arrows);

        Self {
            arrow_cfg: args.arrows,
            sim,
            line_gb,
            point_gb,
        }
    }

    pub fn step(&mut self, field: impl Fn(Vec3x8) -> Vec3x8, dt: f32) {
        self.sim.step(&field, dt);

        self.point_gb.clear();

        self.line_gb.clear();

        particle_mesh(&mut self.point_gb, &self.sim);

        arrow_mesh(&mut self.line_gb, field, &self.arrow_cfg);
    }
}

struct ParticleSim {
    pos: Vec<Vec3x8>,
    vel: Vec<Vec3x8>,
    cfg: ParticleCfg,
}

impl ParticleSim {
    pub fn new(cfg: ParticleCfg) -> Self {
        let mut pos = Vec::with_capacity(cfg.n_parcels);
        let mut vel = Vec::with_capacity(cfg.n_parcels);

        let mut rng = rand::thread_rng();
        for _ in 0..cfg.n_parcels {
            let mut sample = || {
                f32x8::from([(); 8].map(|_| rng.gen_range(-cfg.domain_radius..=cfg.domain_radius)))
            };
            let xyz = Vec3x8::new(sample(), sample(), sample());
            pos.push(xyz);
            vel.push(Vec3x8::zero());
        }

        Self { pos, vel, cfg }
    }

    pub fn step(&mut self, field: impl Fn(Vec3x8) -> Vec3x8, dt: f32) {
        let dt_per_mass = Vec3x8::splat(Vec3::broadcast(dt / self.cfg.mass));
        let dt = Vec3x8::splat(Vec3::broadcast(dt));

        for (pos, vel) in self.pos.iter_mut().zip(&mut self.vel) {
            *vel += field(*pos) * dt_per_mass;
            *pos += *vel * dt;
        }
    }
}

fn particle_mesh(b: &mut GraphicsBuilder, sim: &ParticleSim) {
    for &set in &sim.pos {
        for vert in vec3x8_vertices(set, [1.; 3]) {
            let idx = b.push_vertex(vert);
            b.push_index(idx);
        }
    }
}

fn vec3x8_vertices(pos: Vec3x8, color: [f32; 3]) -> impl Iterator<Item = Vertex> {
    let x: [f32; 8] = pos.x.into();
    let y: [f32; 8] = pos.y.into();
    let z: [f32; 8] = pos.z.into();
    x.into_iter().zip(y).zip(z).map(|((x, y), z)| Vertex {
        pos: [x, y, z],
        color: [1.; 3],
    })
}

fn unit_cube_verts() -> Vec3x8 {
    let mut x = [0f32; 8];
    let mut y = [0f32; 8];
    let mut z = [0f32; 8];

    let bittest = |i: usize, bit: u8| if (i >> bit) & 1 == 0 { 0.0 } else { 1.0 };

    for i in 0..8 {
        x[i] = bittest(i, 0);
        y[i] = bittest(i, 1);
        z[i] = bittest(i, 2);
    }

    Vec3x8::new(x.into(), y.into(), z.into())
}

fn arrow_mesh(b: &mut GraphicsBuilder, field: impl Fn(Vec3x8) -> Vec3x8, cfg: &ArrowCfg) {
    let arrow_sep = 1. / cfg.arrow_density.cbrt();

    let steps = (cfg.arrow_radius * 2. / arrow_sep) as u32;

    let min_xyz = Vec3::broadcast(-cfg.arrow_radius * 2. + arrow_sep);

    let unit = unit_cube_verts() * Vec3x8::splat(Vec3::broadcast(arrow_sep));

    for x in 0..steps {
        for y in 0..steps {
            for z in 0..steps {
                let xyz = Vec3::new(x as f32, y as f32, z as f32);
                let corner = min_xyz + xyz * arrow_sep * 2.;
                let tails = unit + Vec3x8::splat(corner);

                let field_vals = field(tails);

                let tips = tails + field_vals;

                for (tip, tail) in
                    vec3x8_vertices(tips, [1.; 3]).zip(vec3x8_vertices(tails, [0.; 3]))
                {
                    let tail = b.push_vertex(tail);
                    let tip = b.push_vertex(tip);
                    b.push_index(tip);
                    b.push_index(tail);
                }
            }
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct GraphicsBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl GraphicsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a Vertex and return it's index
    pub fn push_vertex(&mut self, v: Vertex) -> u32 {
        let idx: u32 = self
            .vertices
            .len()
            .try_into()
            .expect("Vertex limit exceeded");
        self.vertices.push(v);
        idx
    }

    /// Push an index
    pub fn push_index(&mut self, idx: u32) {
        self.indices.push(idx);
    }

    /// Erase all content
    pub fn clear(&mut self) {
        self.indices.clear();
        self.vertices.clear();
    }

    /// Push the given vertices, and their opposite face
    pub fn push_double_sided(&mut self, indices: &[u32]) {
        self.indices.extend_from_slice(indices);
        self.indices.extend(
            indices
                .chunks_exact(3)
                .map(|face| [face[2], face[1], face[0]])
                .flatten(),
        );
    }
}
