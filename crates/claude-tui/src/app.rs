use tui_input::Input;

#[derive(Clone)]
pub enum Sender {
    User,
    Bot,
}

#[derive(Clone)]
pub struct Message {
    pub sender: Sender,
    pub content: String,
}

pub struct App {
    /// Chat history
    pub messages: Vec<Message>,
    /// User input buffer
    pub input: Input,
    /// Is the bot currently "thinking"?
    pub is_loading: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            messages: vec![
                Message {
                    sender: Sender::Bot,
                    content: "Hello! I am Claude-Rust. Ask me anything.".to_string(),
                }
            ],
            input: Input::default(),
            is_loading: false,
        }
    }

    pub fn append_token(&mut self, token: &str) {
        if let Some(msg) = self.messages.last_mut() {
            if matches!(msg.sender, Sender::Bot) {
                msg.content.push_str(token);
            } else {
                self.messages.push(Message {
                    sender: Sender::Bot,
                    content: token.to_string(),
                });
            }
        }
    }
}
