use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

use wgpu::util::DeviceExt; // brings create_buffer_init into scope

use glam::{Quat, Vec3};

struct State {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    aspect_bind_group: wgpu::BindGroup,
    aspect_buf: wgpu::Buffer,
    light_pos: [f32; 2],
    light_buf: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    orientation: [f32; 4], // quaternion
    orientation_buf: wgpu::Buffer,
    orientation_bind_group: wgpu::BindGroup,
    start_time: Instant,
    time: f32,
    time_buf: wgpu::Buffer,
    time_bind_group: wgpu::BindGroup,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let size = window.inner_size();

        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Raymarcher Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Uniform for aspect ratio
        let aspect = size.width as f32 / size.height as f32;
        let aspect_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Aspect Buffer"),
            contents: bytemuck::cast_slice(&[aspect]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let aspect_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("aspect_bgl"),
            });
        let aspect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &aspect_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: aspect_buf.as_entire_binding(),
            }],
            label: Some("aspect_bg"),
        });

        let light_pos = [0.0, 0.0];
        let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&light_pos),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Light BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buf.as_entire_binding(),
            }],
            label: Some("Light Bind Group"),
        });

        let orientation = [0.0, 0.0, 0.0, 1.0];
        let orientation_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Orientation Buffer"),
            contents: bytemuck::cast_slice(&orientation),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let orientation_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Orientation BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let orientation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &orientation_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: orientation_buf.as_entire_binding(),
            }],
            label: Some("Orientation Bind Group"),
        });

        let start_time = Instant::now();
        let time = start_time.elapsed().as_secs_f32();
        let time_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Time Buffer"),
            contents: bytemuck::cast_slice(&[0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let time_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Time BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let time_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &time_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: time_buf.as_entire_binding(),
            }],
            label: Some("Time Bind Group"),
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&aspect_bind_group_layout, &light_bind_group_layout, &orientation_bind_group_layout, &time_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Raymarcher Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[], // no vertex buffers
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None, // <–– new required field
        });

        let state = State {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            render_pipeline,
            aspect_bind_group,
            aspect_buf,
            light_pos,
            light_buf,
            light_bind_group,
            orientation,
            orientation_buf,
            orientation_bind_group,
            start_time,
            time: time,
            time_buf,
            time_bind_group,
        };

        // Configure surface for the first time
        state.configure_surface();

        state
    }

    fn get_window(&self) -> &Window {
        &self.window
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            // Request compatibility with the sRGB-format texture view we‘re going to create later.
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.configure_surface();

        // Update the aspect ratio uniform buffer
        let new_aspect = self.size.width as f32 / self.size.height as f32;
        self.queue
            .write_buffer(&self.aspect_buf, 0, bytemuck::cast_slice(&[new_aspect]));
    }

    fn render(&mut self) {
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("failed to acquire next swapchain texture");
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sphere Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.aspect_bind_group, &[]);
            rpass.set_bind_group(1, &self.light_bind_group, &[]);
            rpass.set_bind_group(2, &self.orientation_bind_group, &[]);
            rpass.set_bind_group(3, &self.time_bind_group, &[]);
            // Draw 3 vertices → our full-screen triangle
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        surface_texture.present();
    }
}

#[derive(Default)]
pub struct App {
    state: Option<State>,
    keys_held: HashSet<Key>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window object
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone()));
        self.state = Some(state);

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    let light_delta = 0.05;
                    let angle_delta = 0.02;

                    state.time = state.start_time.elapsed().as_secs_f32();

                    let mut orient: Quat = Quat::from_array(state.orientation);

                    if self.keys_held.contains(&Key::Named(NamedKey::ArrowLeft)) {
                        state.light_pos[0] -= light_delta;
                    }
                    if self.keys_held.contains(&Key::Named(NamedKey::ArrowRight)) {
                        state.light_pos[0] += light_delta;
                    }
                    if self.keys_held.contains(&Key::Named(NamedKey::ArrowUp)) {
                        state.light_pos[1] += light_delta;
                    }
                    if self.keys_held.contains(&Key::Named(NamedKey::ArrowDown)) {
                        state.light_pos[1] -= light_delta;
                    }
                    if self.keys_held.contains(&Key::Character("a".into())) {
                        let axis = orient * Vec3::Y;
                        let dq = Quat::from_axis_angle(axis, angle_delta);
                        orient = (dq * orient).normalize();
                    }
                    if self.keys_held.contains(&Key::Character("d".into())) {
                        let axis = orient * Vec3::Y;
                        let dq = Quat::from_axis_angle(axis, -angle_delta);
                        orient = (dq * orient).normalize();
                    }
                    if self.keys_held.contains(&Key::Character("w".into())) {
                        let axis = orient * Vec3::X;
                        let dq = Quat::from_axis_angle(axis, angle_delta);
                        orient = (dq * orient).normalize();
                    }
                    if self.keys_held.contains(&Key::Character("s".into())) {
                        let axis = orient * Vec3::X;
                        let dq = Quat::from_axis_angle(axis, -angle_delta);
                        orient = (dq * orient).normalize();
                    }
                    if self.keys_held.contains(&Key::Character("q".into())) {
                        let axis = orient * Vec3::Z;
                        let dq = Quat::from_axis_angle(axis, angle_delta);
                        orient = (dq * orient).normalize();
                    }
                    if self.keys_held.contains(&Key::Character("e".into())) {
                        let axis = orient * Vec3::Z;
                        let dq = Quat::from_axis_angle(axis, -angle_delta);
                        orient = (dq * orient).normalize();
                    }


                    state.queue.write_buffer(
                        &state.light_buf,
                        0,
                        bytemuck::cast_slice(&[state.light_pos]),
                    );

                    state.orientation = orient.to_array();
                    state.queue.write_buffer(
                        &state.orientation_buf,
                        0,
                        bytemuck::cast_slice(&state.orientation), // see note below
                    );
                    state.queue.write_buffer(
                        &state.time_buf,
                        0,
                        bytemuck::cast_slice(&[state.time]),
                    );

                    state.render();
                    state.get_window().request_redraw(); // loop
                }
            }
            WindowEvent::Resized(size) => {
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                state.resize(size);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: key,
                        state: key_state,
                        ..
                    },
                ..
            } => match key_state {
                ElementState::Pressed => {
                    self.keys_held.insert(key.clone());
                }
                ElementState::Released => {
                    self.keys_held.remove(&key);
                }
            },

            _ => (),
        }
    }
}
