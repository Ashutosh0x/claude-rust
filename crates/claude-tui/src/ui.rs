use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Span, Line},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::app::{App, Sender};

pub fn draw(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Min(1),
                Constraint::Length(3),
            ]
            .as_ref(),
        )
        .split(f.size());

    let (chat_area, input_area) = (chunks[0], chunks[1]);

    // Draw chat history
    let messages: Vec<ListItem> = app
        .messages
        .iter()
        .map(|m| {
            let (prefix, color) = match m.sender {
                Sender::User => ("You: ", Color::Yellow),
                Sender::Bot => ("Claude: ", Color::Cyan),
            };

            let content = vec![Line::from(vec![
                Span::styled(prefix, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::raw(&m.content),
            ])];
            ListItem::new(content)
        })
        .collect();

    let messages = List::new(messages)
        .block(Block::default().borders(Borders::ALL).title("Chat History"))
        .style(Style::default().fg(Color::White));
    
    f.render_widget(messages, chat_area);

    // Draw Input area
    let input = Paragraph::new(app.input.value())
        .style(match app.is_loading {
            true => Style::default().fg(Color::DarkGray),
            false => Style::default().fg(Color::Yellow),
        })
        .block(Block::default().borders(Borders::ALL).title("Input"));
    
    f.render_widget(input, input_area);

    // Set cursor position:
    // Move cursor to (input_area.x + 1 + cursor_position, input_area.y + 1)
    if !app.is_loading {
        f.set_cursor(
            input_area.x + 1 + app.input.cursor() as u16,
            input_area.y + 1,
        );
    }
}
