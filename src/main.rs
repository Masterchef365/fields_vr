use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};
use structopt::StructOpt;
use ultraviolet::{Vec3x8, Vec3, f32x8};
use std::time::Instant;

#[derive(Debug, StructOpt, Default)]
#[structopt(name = "Fields VR", about = "Visualizes fields")]
struct Opt {
    /// Arrow density per unit volume
    #[structopt(short = "d", long, default_value = "1.0")]
    arrow_density: f32,

    // TODO: Use a sphere instead?
    /// Half of the side length of the cube in which arrows are placed.
    #[structopt(short = "r", long, default_value = "1.0")]
    arrow_radius: f32,

    /// Base length of arrows
    #[structopt(short = "l", long, default_value = "0.01")]
    arrow_length: f32,

    /// Number of particles to simulate
    #[structopt(short = "n", long, default_value = "0.01")]
    n_particles: usize,

    /// Particle mass
    #[structopt(short = "n", long, default_value = "0.01")]
    mass: f32,

    /// Visualize with VR
    #[structopt(long)]
    vr: bool,
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

        self.present.step(dt.as_secs_f32());

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
}

impl FieldPresentation {
    pub fn new(args: &Opt, field: impl Fn(Vec3x8) -> Vec3x8) -> Self {
        let mut line_gb = GraphicsBuilder::new();
        let mut point_gb = GraphicsBuilder::new();

        let sim = ParticleSim::new(args.n_particles, args.mass);

        frame_meshes(&mut line_gb, &mut point_gb, &sim, field);

        Self {
            sim,
            line_gb,
            point_gb,
        }
    }

    pub fn step(&mut self, dt: f32) {
        todo!()
    }
}

struct ParticleSim {
    mass: f32,
    pos: Vec<Vec3x8>,
    vel: Vec<Vec3x8>,
}

impl ParticleSim {
    pub fn new(count: usize, mass: f32) -> Self {
        todo!()
    }

    pub fn update(&mut self, field: impl Fn(Vec3x8) -> Vec3x8, dt: f32) {
        let dt_per_mass = Vec3x8::splat(Vec3::broadcast(dt / self.mass));
        let dt = Vec3x8::splat(Vec3::broadcast(dt));

        for (pos, vel) in self.pos.iter_mut().zip(&mut self.vel) {
            *vel += field(*pos) * dt_per_mass;
            *pos += *vel * dt;
        }
    }
}

fn frame_meshes(
    points: &mut GraphicsBuilder,
    lines: &mut GraphicsBuilder,
    sim: &ParticleSim,
    field: impl Fn(Vec3x8) -> Vec3x8,
) {
    particle_mesh(points, sim);
    arrow_mesh(lines, field);
}

fn particle_mesh(b: &mut GraphicsBuilder, sim: &ParticleSim) {
    for set in &sim.pos {
        let k: [f32; 8] = set.x.into();
        //for (x, y, z) in set.x.
    }
}

fn arrow_mesh(b: &mut GraphicsBuilder, field: impl Fn(Vec3x8) -> Vec3x8) {
    todo!()
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
