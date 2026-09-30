import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "YogurtSenseNova.TextPreview",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["YogurtSenseNovaVQA", "YogurtSenseNovaInterleave"].includes(nodeData.name)) return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (!message.text) return;
            if (!this.sensenovaText) {
                const widget = ComfyWidgets.STRING(this, "result", ["STRING", { multiline: true }], app).widget;
                widget.options.serialize = false;
                widget.inputEl.readOnly = true;
                this.sensenovaText = widget;
            }
            this.sensenovaText.value = message.text.join("\n");
            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
