//! Native macOS menu bar (`NSStatusBar`) icon for Colab Playground.
//!
//! Shows a lightning-bolt SF Symbol (`bolt.fill`) in the top-right status bar with quick actions
//! to show/focus the main window, hide it (`Cmd+H`), or quit via the app's billing-safe close path.

#![cfg(target_os = "macos")]

use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel,
};
use objc2_app_kit::{
    NSApplication, NSImage, NSMenu, NSMenuItem, NSStatusBar, NSStatusItem,
    NSVariableStatusItemLength,
};
use objc2_foundation::{NSObject, NSObjectProtocol, NSString};

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "ColabPlaygroundStatusBarTarget"]
    struct StatusBarTarget;

    unsafe impl NSObjectProtocol for StatusBarTarget {}

    impl StatusBarTarget {
        #[unsafe(method(showApp:))]
        fn show_app(&self, _sender: Option<&NSObject>) {
            if let Some(mtm) = MainThreadMarker::new() {
                let app = NSApplication::sharedApplication(mtm);
                app.unhide(None);
                #[allow(deprecated)]
                app.activateIgnoringOtherApps(true);
                let windows = app.windows();
                for i in 0..windows.count() {
                    let win = windows.objectAtIndex(i);
                    win.makeKeyAndOrderFront(None);
                }
            }
        }

        #[unsafe(method(hideApp:))]
        fn hide_app(&self, _sender: Option<&NSObject>) {
            if let Some(mtm) = MainThreadMarker::new() {
                let app = NSApplication::sharedApplication(mtm);
                app.hide(None);
            }
        }

        #[unsafe(method(quitApp:))]
        fn quit_app(&self, _sender: Option<&NSObject>) {
            if let Some(mtm) = MainThreadMarker::new() {
                let app = NSApplication::sharedApplication(mtm);
                app.unhide(None);
                #[allow(deprecated)]
                app.activateIgnoringOtherApps(true);
                let windows = app.windows();
                if windows.count() > 0 {
                    let win = windows.objectAtIndex(0);
                    win.makeKeyAndOrderFront(None);
                    // Triggers RootView's close guard so active GPU billing prompts before exit.
                    win.performClose(None);
                } else {
                    app.terminate(None);
                }
            }
        }
    }
);

impl StatusBarTarget {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}

struct StatusBarState {
    _item: Retained<NSStatusItem>,
    _menu: Retained<NSMenu>,
    _target: Retained<StatusBarTarget>,
}

thread_local! {
    static STATUS_BAR: RefCell<Option<StatusBarState>> = const { RefCell::new(None) };
}

/// Installs the macOS top-right menu bar status item (idempotent; runs on main thread).
pub fn install() {
    let Some(mtm) = MainThreadMarker::new() else {
        log::warn!("status_bar: not on macOS main thread; skipping");
        return;
    };

    STATUS_BAR.with(|slot| {
        if slot.borrow().is_some() {
            return;
        }

        let bar = NSStatusBar::systemStatusBar();
        let item = bar.statusItemWithLength(NSVariableStatusItemLength);

        if let Some(button) = item.button(mtm) {
            let sym = NSString::from_str("bolt.fill");
            let desc = NSString::from_str("Colab Playground");
            if let Some(img) =
                NSImage::imageWithSystemSymbolName_accessibilityDescription(&sym, Some(&desc))
            {
                img.setTemplate(true);
                button.setImage(Some(&img));
            } else {
                let title = NSString::from_str("⚡");
                button.setTitle(&title);
            }
            let tip = NSString::from_str("Colab Playground");
            button.setToolTip(Some(&tip));
        }

        let target = StatusBarTarget::new(mtm);
        let menu = NSMenu::new(mtm);
        let empty = NSString::from_str("");

        unsafe {
            let show_title = NSString::from_str("显示 Colab Playground");
            let show_item = NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                &show_title,
                Some(sel!(showApp:)),
                &empty,
            );
            show_item.setTarget(Some(&target));
            menu.addItem(&show_item);

            let hide_title = NSString::from_str("隐藏窗口 (⌘H)");
            let hide_item = NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                &hide_title,
                Some(sel!(hideApp:)),
                &empty,
            );
            hide_item.setTarget(Some(&target));
            menu.addItem(&hide_item);

            menu.addItem(&NSMenuItem::separatorItem(mtm));

            let quit_title = NSString::from_str("退出 Colab Playground (⌘Q)");
            let quit_item = NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                &quit_title,
                Some(sel!(quitApp:)),
                &empty,
            );
            quit_item.setTarget(Some(&target));
            menu.addItem(&quit_item);
        }

        item.setMenu(Some(&menu));
        *slot.borrow_mut() = Some(StatusBarState {
            _item: item,
            _menu: menu,
            _target: target,
        });
    });
}
