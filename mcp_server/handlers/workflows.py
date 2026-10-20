"""Handler functions for workflow tools."""

from mcp.types import TextContent

from mcp_server.shared import _get_state_db


def handle_workflow_create(arguments: dict) -> list[TextContent]:
    wf_name = arguments.get("name")
    steps = arguments.get("steps")

    if not wf_name:
        return [TextContent(type="text", text="Error: name is required")]
    if not steps:
        return [TextContent(type="text", text="Error: steps is required")]

    client_name = arguments.get("client_name")

    try:
        _, workflow = _get_state_db(client_name)
        workflow_id = workflow.create(wf_name, steps)
        wf = workflow.get(workflow_id)
        next_ready = workflow.next_steps(workflow_id)

        lines = [f"Created workflow **{wf_name}**"]
        lines.append(f"ID: `{workflow_id}`")
        lines.append(f"Steps: {len(wf['steps'])}")
        lines.append("")
        for s in wf["steps"]:
            deps = ""
            if s["depends_on"]:
                dep_names = []
                for dep_id in s["depends_on"]:
                    dep_step = next((st for st in wf["steps"] if st["id"] == dep_id), None)
                    if dep_step:
                        dep_names.append(dep_step["name"])
                deps = f" (after: {', '.join(dep_names)})"
            lines.append(f"  {s['sort_order']+1}. **{s['name']}**{deps}")
            lines.append(f"     ID: `{s['id']}`")
            if s.get("description"):
                lines.append(f"     {s['description']}")

        if next_ready:
            lines.append(f"\nReady to start: {', '.join(s['name'] for s in next_ready)}")

        return [TextContent(type="text", text="\n".join(lines))]
    except ValueError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error creating workflow: {e}")]


def handle_workflow_status(arguments: dict) -> list[TextContent]:
    workflow_id = arguments.get("workflow_id")
    if not workflow_id:
        return [TextContent(type="text", text="Error: workflow_id is required")]

    client_name = arguments.get("client_name")

    try:
        _, workflow = _get_state_db(client_name)
        wf = workflow.get(workflow_id)
        if not wf:
            return [TextContent(type="text", text=f"Workflow `{workflow_id}` not found.")]

        next_ready = workflow.next_steps(workflow_id)

        status_icons = {
            "pending": "[ ]",
            "running": "[~]",
            "completed": "[x]",
            "failed": "[!]",
            "skipped": "[-]",
        }

        lines = [f"**{wf['name']}** ({wf['status']})"]
        lines.append(f"ID: `{wf['id']}`\n")

        for s in wf["steps"]:
            icon = status_icons.get(s["status"], "[ ]")
            lines.append(f"{icon} **{s['name']}** — {s['status']}")
            lines.append(f"    ID: `{s['id']}`")
            if s.get("result"):
                lines.append(f"    Result: {s['result']}")

        if next_ready:
            lines.append(f"\nReady to start: {', '.join(s['name'] for s in next_ready)}")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def handle_workflow_update(arguments: dict) -> list[TextContent]:
    workflow_id = arguments.get("workflow_id")
    step_id = arguments.get("step_id")
    status = arguments.get("status")

    if not workflow_id:
        return [TextContent(type="text", text="Error: workflow_id is required")]
    if not step_id:
        return [TextContent(type="text", text="Error: step_id is required")]
    if not status:
        return [TextContent(type="text", text="Error: status is required")]

    result = arguments.get("result")
    client_name = arguments.get("client_name")

    try:
        _, workflow = _get_state_db(client_name)
        updated = workflow.update_step(workflow_id, step_id, status, result=result)
        if not updated:
            return [TextContent(type="text", text=f"Step `{step_id}` not found in workflow `{workflow_id}`.")]

        wf = workflow.get(workflow_id)
        next_ready = workflow.next_steps(workflow_id)

        text = f"Updated step to **{status}**. Workflow is now **{wf['status']}**."
        if next_ready:
            text += f"\nNext steps ready: {', '.join(s['name'] for s in next_ready)}"

        return [TextContent(type="text", text=text)]
    except ValueError as e:
        return [TextContent(type="text", text=f"Validation error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def handle_workflow_list(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    status_filter = arguments.get("status")

    try:
        _, workflow = _get_state_db(client_name)
        workflows = workflow.list(status=status_filter)

        if not workflows:
            return [TextContent(type="text", text="No workflows found.")]

        lines = [f"**Workflows** ({len(workflows)} total):\n"]
        for wf in workflows:
            progress = f"{wf['completed_steps']}/{wf['total_steps']} steps"
            lines.append(f"- **{wf['name']}** ({wf['status']}) — {progress}")
            lines.append(f"  ID: `{wf['id']}`")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]
