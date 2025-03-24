import os

def generate_script(template_path, output_path, placeholders=None):
    if placeholders is None:
        print("Placeholders is empty!")
        return

    with open(template_path, 'r', encoding='utf-8') as template_file:
        content = template_file.read()


    for placeholder, value in placeholders.items():
        content = content.replace(placeholder, str(value))


    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(content)

    print(f"File generated: {output_path}")


template_file = "scriptTemplates/JulianTest.jwfscript"
new_file = "scripts/TestScript.jwfscript"
placeholders = {
    "PLACEHOLDER_1": 1,
    "PLACEHOLDER_2": 2
}

generate_script(template_file, new_file, placeholders)