use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

struct Engine {
    child: Child,
    input: ChildStdin,
    output: Receiver<Value>,
}
impl Engine {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_hikari_rei"))
            .args(["tbp", "--batch-size", "8", "--max-nodes", "200000"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let stream = child.stdout.take().unwrap();
        let (tx, output) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines() {
                let Ok(line) = line else {
                    break;
                };
                if tx.send(serde_json::from_str(&line).unwrap()).is_err() {
                    break;
                }
            }
        });
        let engine = Self {
            child,
            input,
            output,
        };
        assert_eq!(engine.read()["type"], "info");
        engine
    }
    fn send(&mut self, value: Value) {
        writeln!(self.input, "{value}").unwrap();
        self.input.flush().unwrap();
    }
    fn read(&self) -> Value {
        self.output
            .recv_timeout(Duration::from_secs(10))
            .expect("engine response")
    }
    fn suggest(&mut self) -> Value {
        self.send(json!({"type":"suggest"}));
        let value = self.read();
        assert_eq!(value["type"], "suggestion", "{value}");
        value
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn start_ponders_before_play_and_confirmed_move_retains_graph() {
    let mut engine = Engine::new();
    engine.send(json!({"type":"rules","randomizer":"seven_bag"}));
    assert_eq!(engine.read()["type"], "ready");
    engine.send(json!({"type":"start","position_id":101,"board":[],
        "queue":["T","O","I","J","L","Z","S","I","T","O","J","L"]}));
    let first = engine.suggest();
    assert!(!first["moves"].as_array().unwrap().is_empty());
    std::thread::sleep(Duration::from_millis(200));
    let second = engine.suggest();
    assert_eq!(second["position_id"], 101);
    assert!(
        second["move_info"]["search"]["expanded"].as_u64().unwrap()
            > first["move_info"]["search"]["expanded"].as_u64().unwrap()
    );
    // A suggestion is read-only. The host commits only when placement occurs.
    engine.send(json!({"type":"stop"}));
    let stopped = engine.suggest();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(stopped, engine.suggest());
    // Commit the earlier suggestion even after pondering has changed ranks.
    engine.send(json!({"type":"play","position_id":102,"move":second["moves"][0]}));
    let advanced = engine.suggest();
    assert_eq!(advanced["position_id"], 102);
    assert!(
        advanced["move_info"]["search"]["retained_after_advance"]
            .as_u64()
            .unwrap()
            > 1
    );
    let nodes = advanced["move_info"]["search"]["nodes"].as_u64().unwrap();
    engine.send(json!({"type":"new_piece","piece":"Z"}));
    let revealed = engine.suggest();
    assert_eq!(
        revealed["move_info"]["search"]["nodes"].as_u64().unwrap(),
        nodes
    );
    let mut reset_board = revealed["board"].clone();
    let old_rows = reset_board["rows"].as_array().unwrap().clone();
    for y in (2..24).rev() {
        reset_board["rows"][y] = old_rows[y - 2].clone();
    }
    reset_board["rows"][0] = json!(1022);
    reset_board["rows"][1] = json!(1022);
    engine.send(json!({"type":"reset","position_id":103,"board":reset_board}));
    let reset = engine.suggest();
    assert_eq!(reset["position_id"], 103);
    assert_eq!(reset["queue"], revealed["queue"]);
    assert_eq!(reset["board"]["bag"], revealed["board"]["bag"]);
    assert_eq!(reset["move_info"]["search"]["nodes"], 1);
    // A stopped engine accepts reveals without starting another search.
    engine.send(json!({"type":"quit"}));
    assert!(engine.child.wait().unwrap().success());
}

#[test]
fn same_kind_at_bag_boundary_is_a_representable_normal_placement() {
    let mut engine = Engine::new();
    engine.send(json!({"type":"start","board":[],"bag":1,
        "queue":["I","I","O","T","J","L","S","Z"]}));
    engine.send(json!({"type":"stop"}));
    let suggestion = engine.suggest();
    assert_eq!(suggestion["moves"][0]["location"]["type"], "I");
    assert_eq!(suggestion["move_info"]["chosen"]["hold"], false);
    assert_eq!(suggestion["move_info"]["path"]["hold"], false);
    engine.send(json!({"type":"play","move":suggestion["moves"][0]}));
    let confirmed = engine.suggest();
    assert!(confirmed["board"]["hold"].is_null());
    assert_eq!(confirmed["queue"][0], "I");
    assert_eq!(confirmed["queue"].as_array().unwrap().len(), 7);
    engine.send(json!({"type":"quit"}));
    assert!(engine.child.wait().unwrap().success());
}
