/**
 * Export all annotation measurements from all slides to a single TSV file.
 * Includes: Image, Class, Name (all classes), probability scores.
 * Run after import_classification_all_slides.groovy.
 */

import qupath.lib.gui.measure.ObservableMeasurementTableData

def outputFile = "/Users/antonino/Desktop/MLGlom/results/all_measurements.tsv"
def project    = getProject()
def writer     = new FileWriter(outputFile)
def header     = null

project.getImageList().each { entry ->
    def imageData   = entry.readImageData()
    def imageName   = entry.getImageName()
    def annotations = imageData.getHierarchy().getAnnotationObjects()

    if (annotations.isEmpty()) return

    def table = new ObservableMeasurementTableData()
    table.setImageData(imageData, annotations)

    if (header == null) {
        header = "Image\tClass\tName\t" + table.getMeasurementNames().join("\t")
        writer.write(header + "\n")
    }

    annotations.each { ann ->
        def cls    = ann.getPathClass()?.getName() ?: "Unclassified"
        def name   = ann.getName() ?: ""
        def values = table.getMeasurementNames().collect { m ->
            def v = table.getNumericValue(ann, m)
            Double.isNaN(v) ? "" : String.format("%.4f", v)
        }
        writer.write("${imageName}\t${cls}\t${name}\t${values.join('\t')}\n")
    }
    println "✓ ${annotations.size()} rows — ${imageName}"
}

writer.close()
println "\n✓ Export complete: ${outputFile}"
