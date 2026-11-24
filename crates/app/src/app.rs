use anyhow::Context as _;
use ely_gpui_component::Assets;
use gpui::{
    App, AppContext, Bounds, Entity, TitlebarOptions, WindowBounds, WindowHandle, WindowOptions,
};

#[cfg(target_os = "macos")]
gpui::actions!(colab_playground, [Hide, HideOthers, UnhideAll, Quit]);

use crate::{
    i18n::{keys, tr},
    platform,
    state::{AppState, StateOptions},
    ui::{
        metrics::{WINDOW_MIN_SIZE, WINDOW_SIZE},
        root::RootView,
    },
};

#[cfg(target_os = "macos")]
fn init_macos_menu(state: Entity<AppState>, cx: &mut App) {
    use gpui::{KeyBinding, Menu, MenuItem};

    cx.on_action(|_: &Hide, cx| cx.hide());
    cx.on_action(|_: &HideOthers, cx| cx.hide_other_apps());
    cx.on_action(|_: &UnhideAll, cx| cx.unhide_other_apps());
    cx.on_action(move |_: &Quit, cx| {
        state.update(cx, |s, cx| s.request_quit(cx));
    });
    cx.bind_keys([
        KeyBinding::new("cmd-h", Hide, None),
        KeyBinding::new("cmd-alt-h", HideOthers, None),
        KeyBinding::new("cmd-q", Quit, None),
    ]);
    cx.set_menus(vec![Menu {
        name: "Colab Playground".into(),
        items: vec![
            MenuItem::action("Hide Colab Playground", Hide),
            MenuItem::action("Hide Others", HideOthers),
            MenuItem::action("Show All", UnhideAll),
            MenuItem::separator(),
            MenuItem::action("Quit Colab Playground", Quit),
        ],
        disabled: false,
    }]);
}

/// Runs the app until its last window closes.
pub fn run(options: StateOptions) {
    gpui_platform::application()
        .with_assets(Assets)
        .run(move |cx: &mut App| {
            if let Err(error) = ely_gpui_component::init(cx) {
                log::error!("app: Ely failed to start: {error:#}");
                cx.quit();
                return;
            }
            let state = AppState::init(options, cx);
            #[cfg(target_os = "macos")]
            init_macos_menu(state.clone(), cx);
            if let Err(error) = open_main_window(state, cx) {
                log::error!("app: {error:#}");
                cx.quit();
                return;
            }
            cx.on_window_closed(|cx, _| {
                if cx.windows().is_empty() {
                    cx.quit();
                }
            })
            .detach();
            cx.activate(true);
        });
}

/// Options for the main window: centred, with a minimum size, and no OS title bar (the title
/// is still set for the taskbar and window switcher; the app draws its own strip, see
/// `ui::titlebar`). Decorations, background and traffic lights per OS come from `platform`.
pub fn window_options(cx: &App) -> WindowOptions {
    let bounds = Bounds::centered(None, WINDOW_SIZE, cx);
    WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        window_min_size: Some(WINDOW_MIN_SIZE),
        titlebar: Some(TitlebarOptions {
            title: Some(tr(cx, keys::APP_NAME)),
            appears_transparent: true,
            traffic_light_position: platform::traffic_light_position(cx),
        }),
        window_decorations: platform::window_decorations(),
        window_background: platform::window_background(),
        app_id: Some("colab-playground".into()),
        ..Default::default()
    }
}

pub fn open_main_window(
    state: Entity<AppState>,
    cx: &mut App,
) -> anyhow::Result<WindowHandle<RootView>> {
    let options = window_options(cx);
    cx.open_window(options, |window, cx| {
        window.on_next_frame(|_, _| log::info!("app: first frame drawn"));
        cx.new(|cx| RootView::new(state, window, cx))
    })
    .context("the main window failed to open")
}
